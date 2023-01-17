import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
df = pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")

df.head()
df.hist(figsize=(15, 10))

plt.show()
df.info()
df.isnull().sum()
df.shape
df[df['total_bedrooms'].isnull()]
print(df['total_rooms'].mean())

df['total_bedrooms'].median()
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)
df.isnull().sum()
df.info()
df['ocean_proximity'].value_counts()
enc = preprocessing.LabelEncoder()

df['ocean_proximity']= enc.fit_transform(df['ocean_proximity'])
df['ocean_proximity'].value_counts()

df.head()
df.dtypes
X = df.drop('median_house_value', axis=1).values

y = df['median_house_value']
scaler = preprocessing.StandardScaler()

X = scaler.fit_transform(X)

X.std()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
regr = LinearRegression()

regr.fit(X_train, y_train)
regr.coef_
regr.intercept_
regr.score(X_train, y_train)