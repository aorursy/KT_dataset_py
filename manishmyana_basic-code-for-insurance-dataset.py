# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df= pd.read_csv("../input/insurance.csv")
df.head()

x=df.iloc[:,:5].values

y=df.iloc[:,6].values

print(x[:5])

print(y[:5])
plt.bar(df['smoker'],y,)

plt.xlabel('smoker')

plt.ylabel('Charges')

plt.title('smoker vs charges')

plt.figure()
plt.figure(figsize=(10,6))

sns.boxplot(x='children',y='age',data=df)

plt.show()
plt.figure(figsize=(14,8))

sns.relplot(x='age',y='bmi',data=df)

plt.show()
plt.figure(figsize=(10,6))

sns.countplot(x='smoker',data=df,palette='Blues')

plt.show()
from sklearn.preprocessing import LabelEncoder

label_x= LabelEncoder()

x[:,1]=label_x.fit_transform(x[:,1])

x[:,4]=label_x.fit_transform(x[:,4])

print("The first five row of input variable \n",x[:5])
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.linear_model import LinearRegression

lin_regressor= LinearRegression()

lin_regressor.fit(x_train,y_train)

y_predict=lin_regressor.predict(x_test)

score=lin_regressor.score(x_test,y_test)

print("Linear Regression Accuracy is",score*100)

from sklearn.tree import DecisionTreeRegressor

dec_regressor= DecisionTreeRegressor(criterion='mse',random_state=0)

dec_regressor.fit(x_train,y_train)

score=dec_regressor.score(x_test,y_test)

print("Decesion Tree Regression Accuracy score is ", score*100)
from sklearn.ensemble import RandomForestRegressor

rand_regressor= RandomForestRegressor(n_estimators=10,random_state=0)

rand_regressor.fit(x_train,y_train)

score=rand_regressor.score(x_test,y_test)

print("Random Forest Regression Accuracy score is ", score*100)





