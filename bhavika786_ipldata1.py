# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing data

df = pd.read_excel("../input/IPLdata Sold_Base.xlsx")

df.head()
df.columns
df.dtypes
df.size
df.shape
df.count() #checking null values
#Splitting Dataset into train and test data

features = df.drop(["PLAYER NAME","Sold-Base"],axis="columns")

target=df["Sold-Base"]

x_train,x_test,y_train,y_test=train_test_split(features,target,random_state=0)

print("Dimensionality of Training Data: X",x_train.shape,"Y",y_train.shape)

print("Dimensionality of Test Data: X",x_test.shape,"Y",y_test.shape)

model = LinearRegression()

model.fit(x_train,y_train)
price_predict=model.predict(x_test)
#Intercept and Coefficients

print("Intercept : ",model.intercept_)

print("Coefficients : ",model.coef_)