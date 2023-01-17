import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

import os

os.listdir("../input")


dataset=pd.read_csv("../input/data.csv")

dataset.head()

dataset.isnull().sum()





dataset["Subscribers"]=pd.to_numeric(dataset["Subscribers"],errors="coerce")



dataset["Video Uploads"]=pd.to_numeric(dataset["Video Uploads"],errors="coerce")



dataset=dataset.dropna()


X=dataset.iloc[:,[1,3,4]].values

y=dataset.iloc[:,-1].values


from sklearn.preprocessing import Imputer

imp=Imputer()

X[:,[1,2]]=imp.fit_transform(X[:,[1,2]])



from sklearn.preprocessing import LabelEncoder

lab_x=LabelEncoder()



X[:,0]=lab_x.fit_transform(X[:,0].astype(str))


from sklearn.preprocessing import OneHotEncoder

one=OneHotEncoder(categorical_features=[0])

X=one.fit_transform(X)

X=X.toarray()


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X=sc.fit_transform(X)


pd.scatter_matrix(dataset,alpha=0.4)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)



from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(X_train,y_train)



y_predict=lin_reg.predict(X_test)



lin_reg.score(X_train,y_train)

lin_reg.score(X_test,y_predict)

lin_reg.score(X,y)
lin_reg.coef_