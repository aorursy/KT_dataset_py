
import numpy as np 
import pandas as pd 


import matplotlib.pyplot as plt  
%matplotlib inline

dataset = pd.read_csv("../input/Absenteeism_at_work.csv")  


X = dataset.drop('Absenteeism time in hours', axis=1)  
y = dataset['Absenteeism time in hours']  

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42)  
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test)  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
print("Confusion Matrix: \n")
print(confusion_matrix(y_test, y_pred))
print("\n")
print("Metrics: \n")
print(classification_report(y_test, y_pred))  
print("Accuracy %\n")
print(accuracy_score(y_test, y_pred)*100)