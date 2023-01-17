# Importing libraies

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from math import sqrt
# Importing Data

data = pd.read_csv('../input/student-mat.csv')

data.head()
data.drop(['school','age'], axis=1, inplace=True)



data['GAvg'] = (data['G1'] + data['G2'] + data['G3']) / 3

data.drop(['G1','G2','G3'], axis=1, inplace=True)
d = {'yes':1, 'no':0}

data['schoolsup'] = data['schoolsup'].map(d)

data['famsup'] = data['famsup'].map(d)

data['paid'] = data['paid'].map(d)

data['activities'] = data['activities'].map(d)

data['nursery'] = data['nursery'].map(d)

data['higher'] = data['higher'].map(d)

data['internet'] = data['internet'].map(d)

data['romantic'] = data['romantic'].map(d)



d = {'M':0, 'F':1}

data['sex'] = data['sex'].map(d)



d = {'U': 1, 'R': 0}

data['address'] = data['address'].map(d)



# map the famili size data

d = {'LE3': 1, 'GT3': 0}

data['famsize'] = data['famsize'].map(d)



# map the parent's status

d = {'T': 1, 'A': 0}

data['Pstatus'] = data['Pstatus'].map(d)



# map the parent's job

d = {'teacher': 0, 'health': 1, 'services': 2,'at_home': 3,'other': 4}

data['Mjob'] = data['Mjob'].map(d)

data['Fjob'] = data['Fjob'].map(d)



# map the reason data

d = {'home': 0, 'reputation': 1, 'course': 2,'other': 3}

data['reason'] = data['reason'].map(d)



# map the guardian data

d = {'mother': 0, 'father': 1, 'other': 2}

data['guardian'] = data['guardian'].map(d)



data = pd.get_dummies(data, columns=['guardian','reason','Mjob','Fjob','Pstatus','famsize','address','sex'])
# Creating I and D

I = data.iloc[:, :47].values

I = np.delete(I, 20, axis=1)

D = data.iloc[:, 20].values
I_train, I_test, D_train, D_test = train_test_split(I, D, test_size = 0.2, random_state = 0)
regressor = RandomForestRegressor(n_estimators=100, min_samples_split=16)

regressor.fit(I_train,D_train)
D_pred = regressor.predict(I_test)
plt.scatter(np.arange(0,79),D_test, color='red')

plt.scatter(np.arange(0,79),D_pred, color='blue')

plt.title('Grade Predictions')

plt.plot()
regressor.score(I_test,D_test)