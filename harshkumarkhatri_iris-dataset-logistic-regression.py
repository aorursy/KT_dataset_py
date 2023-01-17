from sklearn import linear_model

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix
dataset=pd.read_csv("/kaggle/input/iris/Iris.csv")

dataset.head()
dataset.isnull().sum()
dataset.info()
y=dataset["Species"]

x=dataset.drop("Species",axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)

x_train.head()
reg=linear_model.LogisticRegression(C=0.01,max_iter=1000)

reg.fit(x_train,y_train)

y_predict=reg.predict(x_test)

accuracy_score(y_predict,y_test)
sns.pairplot(dataset)



sns.pairplot(dataset,hue="Species")
print(y_predict)

y_test.head(100)
confusion_matrix(y_predict,y_test)