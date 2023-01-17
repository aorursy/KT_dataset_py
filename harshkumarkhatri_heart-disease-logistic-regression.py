from sklearn import linear_model

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import train_test_split
dataset=pd.read_csv("/kaggle/input/heart-disease-prediction-using-logistic-regression/framingham.csv")

dataset.head
dataset.isnull().sum()
dataset=dataset.dropna(axis=0)
dataset.isnull().sum()
y=dataset["TenYearCHD"]

x=dataset.drop("TenYearCHD",axis=1)

x.head
reg=linear_model.LogisticRegression()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)

x_train.head(2)

sns.pairplot(dataset,hue="TenYearCHD")
reg.fit(x_train,y_train)

y_predict=reg.predict(x_test)

print(y_predict)

accuracy_score(y_predict,y_test)
confusion_matrix(y_predict,y_test)
