import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df=pd.read_csv('../input/diamonds/diamonds.csv').drop('Unnamed: 0',axis=1)
df
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df['cut']=le.fit_transform(df['cut'])

df['color']=le.fit_transform(df['color'])

df['clarity']=le.fit_transform(df['clarity'])
y=df['price']

X=df.drop('price',axis=1)



sns.distplot(y)
sns.pairplot(X) 
plt.figure(figsize=(10,10))

sns.heatmap(df.corr(),annot=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LinearRegression

from sklearn import linear_model

from sklearn import linear_model

lr=LinearRegression()

lasso = linear_model.Lasso(alpha=0.1)

ridge = linear_model.Lasso(alpha=0.1)



lr.fit(X_train,y_train)

lasso.fit(X_train,y_train)

ridge.fit(X_train,y_train)
print("Linear Regression:",lr.score(X_test,y_test))

print("Lasso Regression:",lasso.score(X_test,y_test))

print("Ridge Regression:",ridge.score(X_test,y_test))
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X=scaler.fit_transform(X)
from sklearn.linear_model import LinearRegression

from sklearn import linear_model

from sklearn import linear_model

lr=LinearRegression()

lasso = linear_model.Lasso(alpha=0.1)

ridge = linear_model.Lasso(alpha=0.1)



lr.fit(X_train,y_train)

lasso.fit(X_train,y_train)

ridge.fit(X_train,y_train)
print("Linear Regression:",lr.score(X_test,y_test))

print("Lasso Regression:",lasso.score(X_test,y_test))

print("Ridge Regression:",ridge.score(X_test,y_test))