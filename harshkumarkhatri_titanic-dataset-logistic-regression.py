from sklearn import linear_model

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import train_test_split
dataset_train=pd.read_csv("/kaggle/input/titanic/train_data.csv")

dataset_train.head()
dataset_train=dataset_train.drop(dataset_train.columns[0],axis=1)
dataset_train.info()
y=dataset_train["Survived"]

x=dataset_train.drop(dataset_train.columns[1],axis=1)

x.info()
x_train,x_test,y_train,y_test=train_test_split(x,y)

y_test.head()
reg=linear_model.LogisticRegression()

reg.fit(x,y)
dataset_test=pd.read_csv("/kaggle/input/titanic/test_data.csv")

dataset_test.head()
dataset_test=dataset_test.drop(dataset_test.columns[0],axis=1)

y=dataset_test["Survived"]
dataset_test.info()
x_test=dataset_test.drop(dataset_test.columns[1],axis=1)

x_test.info()
y_predict=reg.predict(x_test)

print(y_predict)
accuracy_score(y_predict,y)
confusion_matrix(y_predict,y)

import seaborn as sns

sns.pairplot(dataset_train)
# sns.pairplot(dataset_train,hue="Survived")

# This shows correct plots in my pc but is showing an error here.