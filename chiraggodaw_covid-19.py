# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *

from sklearn.impute import SimpleImputer

import tensorflow as tf

from tensorflow import keras

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import Ridge

from sklearn.linear_model import SGDRegressor

from sklearn.svm import SVR

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

test_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()



features = ['Country/Region','Lat', 'Long','Date']#,'ConfirmedCases','Fatalities']

train_data['Country/Region'] = labelencoder.fit_transform(train_data['Country/Region'])

test_data['Country/Region'] = labelencoder.fit_transform(test_data['Country/Region'])



X1 = train_data[features]

X2 = pd.get_dummies(X1)

test_X1 = test_data[features]

test_X2 = pd.get_dummies(test_X1)

X, test_X = X2.align(test_X2,join ='inner', axis= 1)

# my_imputer = SimpleImputer()

# X = my_imputer.fit_transform(X)

# test_X = my_imputer.fit_transform(test_X)
# y = []

# for i in train_data.iterrows():

#     y.append(list(np.array(i)[1][6:]))

# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)  

y_cases = np.array(train_data.ConfirmedCases)

y_fatalities = np.array(train_data.Fatalities)
# print(train_X.shape)

# train_y = np.array(train_y)

# val_y = np.array(val_y)

# print(train_y.shape)
# m =1000

# for  i in range(2,10000):

#     model = RandomForestRegressor(n_estimators = 4, max_leaf_nodes = i, random_state = 1)

#     model.fit(train_X, train_y)

#     val_predictions = model.predict(val_X)

#     if(m > mean_absolute_error(val_predictions, val_y)):

#         max_leaf_node = i

#         m = mean_absolute_error(val_predictions, val_y)

#         print(str(i) + " --> " + str(m))



# print(m)
model_cases = RandomForestRegressor(max_leaf_nodes = 999, n_estimators = 4)

model_cases.fit(X, y_cases)



model_fatalities = RandomForestRegressor(max_leaf_nodes = 999, n_estimators = 4)

model_fatalities.fit(X, y_fatalities)
prediction = model_fatalities.predict(X)

print(mean_absolute_error(prediction,y_fatalities))
predictions_cases = model_cases.predict(test_X)

predictions_fatalities = model_fatalities.predict(test_X)

data = pd.DataFrame()

data.insert(0,'ForecastId',np.array(range(1,12213)))

data.insert(1,'ConfirmedCases',predictions_cases)

data.insert(2,'Fatalities',predictions_fatalities)



print(data)
data.to_csv('submission.csv', index=False)
