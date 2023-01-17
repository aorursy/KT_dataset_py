# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = "../input/IPLdata Sold_Base.xlsx"

df = pd.read_excel("../input/ipl-data/IPLdata Sold_Base.xlsx")

df.head()
df.describe()
df.dtypes
df.columns
df.shape
df.isnull().sum()
target=df["Sold-Base"]

features=df.drop(["Sold-Base","PLAYER NAME"],axis="columns")

x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=0)

print("Shape of X train dataset:",x_train.shape," and Y train dataset:",y_train.shape)

print("Shape of X test dataset:",x_test.shape," and Y test dataset:",y_test.shape)
model = LinearRegression()

model.fit(x_train,y_train)

y_predict=model.predict(x_test)
intercept = model.intercept_

intercept
coefficients=model.coef_

coefficients
a=pd.DataFrame(features.columns,columns=["Attributes"])

b=pd.DataFrame(coefficients,columns=["Coefficients"])

coeff_df = pd.concat([a,b],axis=1)

coeff_df
print("Mean Absolute Error:",metrics.mean_absolute_error(y_test,y_predict))

print("Mean Squared Error:",metrics.mean_squared_error(y_test,y_predict))

print("Root Mean Squared Error:",np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
#Doubt

a = pd.DataFrame(y_predict,columns=["Predicted Values"])

b = pd.DataFrame(y_test,columns=["Actual Values"]).transpose()
x=df["Sold-Base"].head()

y=df["PLAYER NAME"].head()

plt.bar(x=df["PLAYER NAME"].head(),y=df["Sold-Base"].head(),color="blue",height=df["Sold-Base"].max)

plt.show()