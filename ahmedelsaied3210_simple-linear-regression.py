# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load libraries

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
#load dataset

dataset=pd.read_csv('../input/heights-and-weights/data.csv')

dataset.head()
# check null values

print(dataset.isnull().sum())
# split dataset to X and Y

x=dataset.iloc[:,:-1].values

y=dataset.iloc[:,-1].values

print(x.shape)

print(y.shape)
#split dataset

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)
#build model using LinearRegression model

from sklearn.linear_model import LinearRegression

LR=LinearRegression()

LR.fit(x_train,y_train)
# predict x_test at the model

LR_predict=LR.predict(x_test)



# Visualising the model

plt.scatter(x_train,y_train,color='red')

plt.plot(x_train,LR.predict(x_train),color='blue')

plt.xlabel('Height')

plt.ylabel('Weight')

plt.show()
# Visualising the Test set results

plt.scatter(x_test,y_test,color='red')

plt.plot(x_train,LR.predict(x_train),color='blue')

plt.xlabel('Height')

plt.ylabel('Weight')

plt.show()
from sklearn.metrics import mean_squared_error



# Calculation of Mean Squared Error (MSE) 

mean_squared_error(y_test,LR_predict) 
from sklearn.ensemble import RandomForestRegressor



rfr=RandomForestRegressor(n_estimators=100,random_state=42)



rfr.fit(x_train,y_train)
rfr_predection=rfr.predict(x_test)



mean_squared_error(rfr_predection,y_test)