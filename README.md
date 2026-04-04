# KNN
#K-neibors

# All depanded libraies
# KNN Classification Example with scikit-learn
#Import all dependies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

#Load the datsets
iris=load_iris()
x=iris.data
y=iris.target

#train Test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Knn model
k=7
knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(y_pred)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
