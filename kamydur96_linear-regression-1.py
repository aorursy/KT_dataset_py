import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/california-housing-prices/housing.csv')

data.head(3)
data.info()
data['total_bedrooms'].isnull().sum()
data['total_bedrooms'][data['total_bedrooms'].isnull()] = np.mean(data['total_bedrooms'])
data.info()
data['avg_rooms'] = data['total_rooms'] / data['households']

data['avg_bedrooms'] = data['total_bedrooms'] / data['households']

data.head(3)
data.corr()
data['popu_per_house'] = data['population'] / data['households']
data.head(3)
data['ocean_proximity'].unique()
data['NEAR BAY'] = 0

data['<1H OCEAN'] = 0

data['INLAND'] = 0

data['NEAR OCEAN'] = 0

data['ISLAND'] = 0

data.head(2)
data.loc[data['ocean_proximity'] == 'NEAR BAY','NEAR BAY'] = 1

data.loc[data['ocean_proximity'] == '<1H OCEAN','<1H OCEAN'] = 1

data.loc[data['ocean_proximity'] == 'INLAND','INLAND'] = 1

data.loc[data['ocean_proximity'] == 'ISLAND','ISLAND'] = 1
data.head(3)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
X = data.drop(['total_rooms','total_bedrooms','households','ocean_proximity','median_house_value'],axis = 1)

y = data['median_house_value']

print(X.shape)

print(y.shape)
train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.2)

print(train_X.shape)

print(train_y.shape)

lnr_clf = LinearRegression()

lnr_clf.fit(np.array(train_X),train_y)

import math

def roundUp(x):

    return int(math.ceil(x/100))*100
pred = list(map(roundUp,lnr_clf.predict(test_X)))
print(pred[:5])

print(test_y[:5])
from sklearn.metrics import mean_squared_error



prediction = lnr_clf.predict(test_X)

mse = mean_squared_error(test_y,prediction)

rmse = np.sqrt(mse)

print(rmse)