import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("../input/iris.csv")
data.head()
data.isnull().sum()
data.shape
data1=data.dropna(how="any",axis=0)
data1.shape
x=data1.drop(["Species"],axis=1)
y=data1["Species"]
x[0:5]
y[0:5]
corr=data1.corr()
corr
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
model=GaussianNB()
data2=model.fit(x_train,y_train)
data2
predict=model.predict(x_test)
predict
data3=data2.score(x_test,y_test)
data3
