# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_absolute_error

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/weight-height.csv")
df.head(5)
df.info()
df.describe()
df.duplicated().sum()
df.isna().sum()
import matplotlib.pyplot as plt

import seaborn as sns

ax = sns.boxplot(df['Height'])

ax.set_title('Dispersion of Height')

plt.show(ax)
import seaborn as sns

ax = sns.boxplot(df['Weight'])

ax.set_title('Dispersion of Weight')

plt.show(ax)
plt.figure(figsize=(10,7))

ax = sns.scatterplot(x='Height',y='Weight', hue=df['Gender'],style = df['Gender'],size = df['Gender'], data=df)

ax.set_title("Height vs Weight by Gender")

plt.xlabel("Height ")

plt.ylabel("Weight")

plt.show(ax)
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

df.iloc[:,0] = labelencoder.fit_transform(df.iloc[:,0])
df.head(5)
df.corr()["Weight"]
x = df[['Gender','Height']]

y = df['Weight']

#train_test_split() to split the dataset into train and test set at random.

#test size data set should be 30% data

X_train,X_test,Y_train, Y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#Creating an linear regression model object

model = LinearRegression()

#Training the model using training data set

model.fit(X_train, Y_train) 

print("Intercept value:", model.intercept_)

print("Coefficient values:", model.coef_)

Y_train_predict = model.predict(X_train)

Y_train_predict[0:5]



Y_test_predict = model.predict(X_test)
ax = sns.scatterplot(Y_train,Y_train_predict)

ax.set_title("Actual Weight vs Predicted Weight")

plt.xlabel("Actual Weight")

plt.ylabel("Predicted Weight")

plt.show(ax)
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

print("MSE:",np.sqrt(mean_squared_error(Y_train, Y_train_predict)))

print("R-squared value:",round(r2_score(Y_train, Y_train_predict),3))

print("Mean absolute error:",mean_absolute_error(Y_train, Y_train_predict))
