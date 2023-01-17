import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



print(os.listdir("../input"))
df = pd.read_csv("../input/data.csv")

df.head(5)
df.isnull().sum()
plt.title("American Women")

plt.xlabel("Height")

plt.ylabel("Weight")

plt.scatter(df.Height,df.Weight,color='blue')
x = df[['Height']]

y = df['Weight']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

print('x_train shape:', x_train.shape)

print('y_train shape', y_train.shape)

print('x_test shape:', x_test.shape)

print('y_test shape', y_test.shape)

print('percent in x_train:', x_train.shape[0]/(x_train.shape[0] + x_test.shape[0]))

print('percent in x_test:', x_test.shape[0]/(x_train.shape[0] + x_test.shape[0]))
model = LinearRegression()

model.fit(x_train,y_train)
model.score(x_test,y_test)
df.iloc[8]
print(model.predict([[1.68]]))