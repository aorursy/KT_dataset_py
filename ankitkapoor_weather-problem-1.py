# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

#path of the file

data = "../input/WeatherData.csv"



WD = pd.read_csv(data)



# Any results you write to the current directory are saved as output.
# Creating the Prediction variable

y = WD.actual



# Reviewing the columns in data

WD.describe()



#choosing prediction features, X

features = ['year','month','day','temp_2','temp_1', 'average']

X = WD[features]

#print(X)



#Spliting data into training and validation 

train_X, val_X, val_x, val_y = train_test_split(X, y, random_state = 1)



# Defining the model

temp_model = RandomForestRegressor(random_state = 1)

#temp_model = DecisionTreeRegressor



# Fitting the model

temp_model.fit(train_X, train_y)
temp_pred = temp_model.predict(val_X)

val_mae = mean_absolute_error(temp_pred, val_y)

#print(temp_pred) 

#print(val_y)

print(val_mae)

#calculate errors

errors = abs (temp_pred-val_y)

#print (errors)



# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / val_y)

#print(mape)



# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')