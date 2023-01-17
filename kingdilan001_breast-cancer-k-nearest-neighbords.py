import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import math

import seaborn as sns

from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import train_test_split

%matplotlib inline
data = pd.read_csv("../input/cancer/dataR2.csv")

data.head()
data.info()
x = data.drop(["Classification"], axis = 1)

x.head()
y = data["Classification"]

y.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=123)
y_test
from sklearn.preprocessing import StandardScaler  

scaler = StandardScaler()  

scaler.fit(x_train)



x_train = scaler.transform(x_train)  

x_test = scaler.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier

#membuat fungsi klasifikasi KNN

classifier = KNeighborsClassifier(n_neighbors=5)
# Memasukkan data training pada fungsi klasifikasi KNN

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

y_pred
classifier.predict_proba(x_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
# Merapikan hasil confusion matrix

y_actual = pd.Series([1,2,2,2,1,2,2,2,2,1,1,1,1,1,1,1,2,2,2,2,2,1,2,2], name = "actual")

y_pred = pd.Series([1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1,

       1, 2], name = "prediction")

df_confusion = pd.crosstab(y_actual, y_pred)
df_confusion
print(classification_report(y_test, y_pred))