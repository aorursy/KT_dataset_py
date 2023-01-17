import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

RANDOM_STATE = 31415
# metric to optimize
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

scorer = make_scorer(lambda y_test, predictions: np.sqrt(mean_squared_error(y_test, predictions)))
training_set = pd.read_csv('../input/train.csv')
training_set.head()
training_set.plot(x = 'datetime', y = 'casual')
training_set.plot(x = 'datetime', y = 'registered')
corr = training_set.corr()
fig, ax = plt.subplots(figsize=(30, 30))
ax.matshow(corr)

for (i, j), z in np.ndenumerate(corr):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

plt.xticks(range(len(corr.columns)), corr.columns);
plt.yticks(range(len(corr.columns)), corr.columns);
from sklearn.model_selection import train_test_split

# Basic preprocessing which applies to all regression techniques (dependent variable: casual)
data = training_set.drop(columns = ['datetime', 'atemp', 'registered', 'count'])

X_train, X_test, y_train, y_test = train_test_split(data, data.casual, test_size=0.2, random_state = RANDOM_STATE)
X_train = X_train.drop(columns = ['casual'])
X_test = X_test.drop(columns = ['casual'])
# Preprocessing for linear regression

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

one_hot = OneHotEncoder(categorical_features = [0, 1, 2, 3]) #season, holiday, workingday and weather
X_train_norm = one_hot.fit_transform(X_train_norm)
X_test_norm = one_hot.transform(X_test_norm)
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
casual_model = Lasso()
scores = cross_val_score(casual_model, X_train_norm, y_train, cv=5, scoring = scorer)
scores
casual_model.fit(X_train_norm, y_train)
# Same thing for the second variable
# Basic preprocessing which applies to all regression techniques (dependent variable: casual)
data = training_set.drop(columns = ['datetime', 'atemp', 'casual', 'count'])

X_train, X_test, y_train, y_test = train_test_split(data, data.registered, test_size=0.2, random_state = RANDOM_STATE)
X_train = X_train.drop(columns = ['registered'])
X_test = X_test.drop(columns = ['registered'])
# Preprocessing for linear regression

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

one_hot = OneHotEncoder(categorical_features = [0, 1, 2, 3]) #season, holiday, workingday and weather
X_train_norm = one_hot.fit_transform(X_train_norm)
X_test_norm = one_hot.transform(X_test_norm)
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
registered_model = Lasso()
scores = cross_val_score(registered_model, X_train_norm, y_train, cv=5, scoring = scorer)
scores
registered_model.fit(X_train_norm, y_train)
# Final prediction of the baseline models, as I am not going to tweak them, I will move directly to the test data

test_dataset = pd.read_csv("../input/test.csv")
test_data = test_dataset.drop(columns = ['datetime', 'atemp'])
test_data = scaler.transform(test_data)
test_data = one_hot.transform(test_data)
casual = casual_model.predict(test_data)
registered = registered_model.predict(test_data)
total = casual + registered
test_dataset['count'] = pd.Series(total)
test_dataset[test_dataset['count'] < 0]
test_dataset.loc[test_dataset['count'] < 0, 'count'] = 0
test_dataset[test_dataset['count'] <= 0]
test_dataset[['datetime', 'count']].to_csv('result.csv', index = False)
