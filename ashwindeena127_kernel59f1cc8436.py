import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/heights-and-weights/data.csv')
df
df.info()
df.describe()
df.isnull().sum()
sns.pairplot(df)
plt.scatter(x= df['Height'],y=df['Weight'],c='red',marker = '*')
df
X = df[['Height']]
y = df['Weight']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
X_train.shape
y_train.shape
y_pred = lr.predict(X_test)
plt.scatter(x=X_test,y=y_test)

plt.plot(X_test,y_pred,c='red')
lr.score(X_test,y_test)
lr.predict([[1.47]])
df.iloc[8]
lr.predict([[1.68]])