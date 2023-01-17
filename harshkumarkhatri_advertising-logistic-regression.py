from sklearn import linear_model

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
dataset=pd.read_csv("/kaggle/input/advertising/advertising.csv")

dataset.head
dataset.isnull().sum()
sns.pairplot(dataset)
sns.pairplot(dataset,hue="Clicked on Ad")
y=dataset["Clicked on Ad"]

x=dataset.drop(["City","Ad Topic Line","Country","Timestamp"],axis=1)

x.head
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10)

x_test.head
reg=linear_model.LogisticRegression(C=0.01)

reg.fit(x_train,y_train)

from sklearn.metrics import accuracy_score,confusion_matrix

y_predict=reg.predict(x_test)

confusion_matrix(y_predict,y_test)
accuracy_score(y_test,y_predict)
# Using standard scaler to add standardariztion to the data

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train.head
x_train=sc.fit_transform(x_train)

x_train
x_test=sc.fit_transform(x_test)

x_test
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)

confusion_matrix(y_test,y_predict)

# As you can see that after standardization we can increase our correct predicting values a bit

accuracy_score(y_test,y_predict)

