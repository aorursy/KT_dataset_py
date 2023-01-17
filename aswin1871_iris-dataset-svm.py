import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

iris = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
iris
iris.describe()
from sklearn.model_selection import train_test_split
x = iris.drop('species',axis=1)

y = iris['species']
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3,random_state=101)
x_train.shape , y_train.shape
x_test.shape , y_test.shape
from sklearn.svm import SVC
model = SVC()
model.fit(x_train,y_train)
pred = model.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))

print("\n")

print(classification_report(y_test,pred))