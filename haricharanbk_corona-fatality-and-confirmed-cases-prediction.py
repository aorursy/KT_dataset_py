# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sbn

from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
train_data.head()
train_data.describe()
train_data.shape
train_data.describe(include='all')
for i in train_data.columns:

    print(train_data[i].value_counts())
plt.figure(figsize=(20,6))

sbn.barplot(x='Province_State', y='ConfirmedCases', data=train_data)

plt.ylim(0, 3000)

plt.xticks(rotation=90)



plt.figure(figsize=(20,6))

sbn.barplot(x='Province_State', y='Fatalities', data=train_data)

plt.ylim(0, 100)

plt.xticks(rotation=90)

plt.figure(figsize=(20,6))

sbn.barplot(x='Country_Region', y='ConfirmedCases', data=train_data)

plt.ylim(0, 3000)

plt.xticks(rotation=90)



plt.figure(figsize=(20,6))

sbn.barplot(x='Country_Region', y='Fatalities', data=train_data)

plt.ylim(0, 100)

plt.xticks(rotation=90)

plt.figure(figsize=(20,6))

sbn.barplot(x='Date', y='ConfirmedCases', data=train_data)

plt.ylim(0, 3000)

plt.xticks(rotation=90)



plt.figure(figsize=(20,6))

sbn.barplot(x='Date', y='Fatalities', data=train_data)

# plt.ylim(0, 160)

plt.xticks(rotation=90)
(train_data['Fatalities'].sum() / train_data['ConfirmedCases'].sum()) * 100
train_data['Date'] = pd.to_datetime(train_data['Date']).astype(int)/ 10**9
train_data = pd.get_dummies(train_data, columns=['Country_Region', 'Province_State'], dummy_na=True)
X = train_data.drop(['Id', 'ConfirmedCases', 'Fatalities'], axis = 1)

Y = train_data[['Fatalities', 'ConfirmedCases']]
def preprocessor(data):

    ids = data['ForecastId']

    frame = pd.get_dummies(data, columns = ['Country_Region', 'Province_State'], dummy_na = True).drop(['ForecastId'], axis = 1)

    frame['Date'] = pd.to_datetime(frame['Date']).astype(int)/ 10**9

    return (ids, frame)

ids, test_x = preprocessor(pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv'))
print(X.shape)

print(Y.shape)

print(test_x.shape)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error



train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size = .1)
print(train_x.shape)

print(train_y.shape)
# model = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# model.fit(train_x, train_y)

import pickle

filename = '/kaggle/input/corona-random-forest-model/random_forest_model.sav'

model = pickle.load(open(filename, 'rb'))
predicted = model.predict(valid_x)

predicted = predicted.round()

rmse = np.sqrt(mean_squared_error(predicted, valid_y))

mae = mean_absolute_error(predicted, valid_y)

print(rmse, mae)
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')

output = pd.DataFrame(columns = submission.columns)

output['ForecastId'] = ids

predicted = model.predict(test_x)

predicted = predicted.round()

output[['ConfirmedCases', 'Fatalities']] = predicted

print(output)
# import pickle

# filename = 'random_forest_model.sav'

# pickle.dump(model, open(filename, 'wb'))

 

# some time later...

 

# load the model from disk

# loaded_model = pickle.load(open(filename, 'rb'))

# result = loaded_model.score(X_test, Y_test)

# print(result)
# import sklearn

# help(sklearn.metrics)



output.to_csv('submission.csv', index = False)