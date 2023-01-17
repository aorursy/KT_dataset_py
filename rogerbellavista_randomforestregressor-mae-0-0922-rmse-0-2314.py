# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import uuid

import pickle



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

import math

import sklearn.metrics as metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# DataSet Read

df = pd.read_csv("../input/MiningProcess_Flotation_Plant_Database.csv",decimal=",")

# Delete date column

df= df.drop(df.columns[[0]], axis=1)

df.head()
# split 80% of training 20% of test

train, test = train_test_split(df, test_size=0.2)

print ('# train:',len(train))

print ('# test: ',len(test))
# Extract column to predict on train X

X = train.drop('% Silica Concentrate', axis=1)

X.head(20)
# Create train y

y = train['% Silica Concentrate']

y.head(20)
# Model creation and fit

model = RandomForestRegressor()

model.fit(X,y)
# Show model results of training set

y_hat = model.predict(X)

mae = metrics.mean_absolute_error(y,y_hat)

mse = metrics.mean_squared_error(y,y_hat)

print ("TRAINING SET")

print ("============")

print ("MAE:                ", mae)

print ("RMSE:               ", math.sqrt(mse))

print ("r2:                 ", model.score(X,y))

print ("feature_importances:",model.feature_importances_)

print ("n_features:         ",model.n_features_)

print ("n_outputs:          ",model.n_outputs_)

print ("last column (% Iron Concentrate) is the highest feature_importances")
# Use test and show results of test set

X= test.drop('% Silica Concentrate', axis=1)

y= test['% Silica Concentrate']

y_hat = model.predict(X)

mae = metrics.mean_absolute_error(y,y_hat)

mse = metrics.mean_squared_error(y,y_hat)

print ("TEST SET")

print ("========")

print ("MAE:                ", mae)

print ("RMSE:               ", math.sqrt(mse))

print ("r2:                 ", model.score(X,y))
print ("Repeat without % Iron Concentrate")
# Extract columnson train X

X2= train.drop(['% Silica Concentrate','% Iron Concentrate'], axis=1)

y2= train['% Silica Concentrate']

# Model creation and fit

model2 = RandomForestRegressor()

model2.fit(X2,y2)

# Show model results of training set

y2_hat = model2.predict(X2)

mae = metrics.mean_absolute_error(y2,y2_hat)

mse = metrics.mean_squared_error(y2,y2_hat)

print ("TRAINING SET( without % Iron Concentrate)")

print ("=========================================")

print ("MAE:                ", mae)

print ("RMSE:               ", math.sqrt(mse))

print ("r2:                 ", model2.score(X2,y2))

print ("feature_importances:",model2.feature_importances_)

print ("n_features:         ",model2.n_features_)

print ("n_outputs:          ",model2.n_outputs_)

# Use test and show results of test set

X2= test.drop(['% Silica Concentrate','% Iron Concentrate'], axis=1)

y2= test['% Silica Concentrate']

y2_hat = model2.predict(X2)

mae = metrics.mean_absolute_error(y2,y2_hat)

mse = metrics.mean_squared_error(y2,y2_hat)

print ("TEST SET( without % Iron Concentrate)")

print ("=====================================")

print ("MAE:                ", mae)

print ("RMSE:               ", math.sqrt(mse))

print ("r2:                 ", model2.score(X2,y2))
