import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
df = pd.read_csv("../input/Mall_Customers.csv")

df.head()
df.describe()
df.info()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])
df.head()
df  = df.drop(columns=['CustomerID'],axis=1)

df.head()
df.plot()
df.hist()
df.head()
X = df.iloc[:,0:-1].values

print(X[0:5,:])
Y = df.iloc[:,3].values

print(Y[0:5])
from sklearn.model_selection import train_test_split

X_train,x_test,Y_train,y_test = train_test_split(X,Y)

print(X_train[0:5,:])
print(Y_train[0:5])
from sklearn.linear_model import LinearRegression 

classifier = LinearRegression()

classifier.fit(X_train,Y_train)
classifier.score(x_test,y_test)
from sklearn.model_selection import cross_val_score

results = cross_val_score(classifier,X,Y,cv=5)

print(results)