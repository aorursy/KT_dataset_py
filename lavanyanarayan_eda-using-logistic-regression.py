import pandas as pd

import numpy as n

import matplotlib.pyplot as ma

import seaborn as se
df=pd.read_csv('heart.csv')

df.head()
df.isnull().sum()
df.groupby('target').mean()
df=df.drop(['thal','fbs','trestbps'],1)

df.head()
from sklearn.model_selection import train_test_split
n.random.seed(21)

x_train,x_test,y_train,y_test=train_test_split(df[['age','sex','cp','chol','restecg','thalach','exang','oldpeak','slope','ca']],df.target,train_size=0.77,random_state=123)
x_train.head()
from sklearn.linear_model import LogisticRegression
mo=LogisticRegression()

mo.fit(x_train,y_train)
mo.score(x_test,y_test)
mo.predict_proba(x_test)