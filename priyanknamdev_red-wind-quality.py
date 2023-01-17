# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print(filenames)

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')



columnsInput = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 

           'density', 'pH', 'sulphates', 'alcohol']

columnTarget = ['quality']



X = dataset[columnsInput]

y = dataset[columnTarget]
train_X, val_X, train_Y, val_Y = train_test_split(X, y, random_state=1)

print(X.shape)

print(y.shape)

print(train_X.shape)

print(train_Y.shape)

print(val_X.shape)

print(val_Y.shape)

# print(X.isnull().sum() & y.isnull().sum() & train_X.isnull().sum() & train_Y.isnull().sum() & val_X.isnull().sum() & val_X.isnull().sum())
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

model = RandomForestRegressor(random_state=1)

model.fit(train_X, train_Y)

model_val = model.predict(val_X)



# print(model_val)



model_mae = mean_absolute_error(model_val, val_Y)

# print(model_mae)

print("Validation MAE for Random Forest Model: {}".format(model_mae))
output = pd.DataFrame({'Quality': model_val})

output.to_csv('submission.csv', index=False)