import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



%matplotlib inline


df=pd.read_csv('../input/Churn_6SIJGngxq2.csv')

df.head(10)

df.iloc[:,-1]

len(df)

df.describe()

#Count is different as it excludes all the data entries which are empty.
df.fillna(value=df.median(), inplace=True)

df.describe()
X=df.copy()

y=df.churned

del X['churned']

print(y)


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3)


df.fillna(value=df.median(), inplace=True)

k=KNeighborsClassifier(n_neighbors=5)

print(k)

k.fit(X, y)
y_pred=k.predict(X)

print(y_pred)
acc=accuracy_score(y,y_pred)

print(acc)
for i in [1,2,3,4,5,6,7,8,9,10]:

    k=KNeighborsClassifier(n_neighbors=i)

    print(k)

    k.fit(X, y)

    y_pred=k.predict(X)

    acc=accuracy_score(y,y_pred)

    print(acc)




