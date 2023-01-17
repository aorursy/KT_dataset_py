# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from sklearn.model_selection import train_test_split

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/HousePrices_HalfMil.csv")

df.head()

df.describe()

y = df.iloc[:,-1]   #Taking last column from dataframe to set output to y

X = df.iloc[:, :-1] #Here taking columns from dataframe except last one (y) 



X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.33, random_state=42) #Splitting dataframe train and test



lr = LinearRegression().fit(X_train, y_train) #Fitting trainin data to Linear Regression Model



predicted_house_prices = lr.predict(X_test) #Predicting target variable 

print("Coefficients: \n",lr.coef_)

print("Mean squared error: %.2f"% mean_squared_error(y_test, predicted_house_prices))

      




