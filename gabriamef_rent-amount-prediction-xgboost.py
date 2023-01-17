# import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from numpy import mean



color = sns.color_palette()

sns.set_style('darkgrid')





from scipy import stats

from scipy.stats import norm, skew



data = pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent_v2.csv')

data.head()

data = data.rename(columns={"hoa (R$)":"hoa", "rent amount (R$)":"rent amount",

                            "property tax (R$)": "property tax", "fire insurance (R$)":"fire insurance",

                            "total (R$)":"total price"})





data['floor'].mask(data['floor'] == '-', '0', inplace=True)

data['floor'] = data['floor'].astype('int')

data.head()
print(data.isnull().sum())
fig, axs = plt.subplots(ncols=3, figsize=(15,5))

sns.boxplot(x=data['area'], ax = axs[0])

sns.boxplot(x=data['rent amount'], ax = axs[1])

sns.boxplot(x=data['floor'], ax = axs[2])

plt.show()
Q1 = data.quantile(0.25)

Q3 = data.quantile(0.75)

IQR = Q3- Q1

print(IQR, data.shape)
data = data[~((data < (Q1 - 5*IQR)) | (data > (Q3 + 5*IQR))).any(axis=1)]

data.shape
fig, axs = plt.subplots(ncols=3, figsize=(10,5))

sns.boxplot(x=data['area'], ax = axs[0])

sns.boxplot(x=data['rent amount'], ax = axs[1])

sns.boxplot(x=data['floor'], ax = axs[2])

plt.show()
plt.figure(figsize=(20,10))

c = data.corr()

sns.heatmap(c,cmap='Blues', annot=True)
from numpy import mean

plt.figure(figsize = (8,4))

sns.barplot(data = data, x='city', y='rent amount', estimator=mean, palette = "Blues")

plt.ylabel('Avg Rent Amount')

plt.xlabel('')

plt.title('Average Rent Amount by City')

plt.legend('')



plt.figure(figsize = (8,4))

sns.barplot(x='city', y='rent amount', data = data, hue = 'animal', palette='Blues', estimator = mean)

plt.ylabel('Avg Rent Amount')

plt.title('Average Rent Amount by City and by Animal Acceptance')



plt.figure(figsize = (8,4)) 

sns.barplot(x='parking spaces', y='rent amount', data = data, palette='Blues', estimator=mean)

plt.ylabel('Avg Rent Amount')

plt.title('Averace Rent Amount by n. of Parking Spaces')

plt.show()



data["city"] = data["city"].astype('category')

data["animal"] = data["animal"].astype('category')

cleanup = {"city": {"São Paulo" : 1, "Porto Alegre" : 2, "Rio de Janeiro": 3, "Campinas":4, "Belo Horizonte":5},

          "animal": {"not acept":0, "acept":1},

          "furniture": {"not furnished":0, "furnished":1}}

data.replace(cleanup, inplace=True)

data.head()
data = pd.get_dummies(data, columns=['city'])

data = data.drop(columns=['city_1'])
data = data[['city_2', 'city_3', 'city_4', 'city_5', 'area', 'rooms','bathroom', 'parking spaces', 'floor', 'furniture', 'hoa', 'property tax', 'fire insurance', 'total price', 'animal', 'rent amount']]

data.head()
col = ['city_2', 'city_3', 'city_4', 'city_5', 'area', 'rooms','bathroom', 'parking spaces', 'floor', 'furniture', 'hoa', 'property tax', 'total price', 'animal']

sns.pairplot(data[col], size=2)

plt.show
data['RentAmountLog'] = np.log(data['rent amount'])

data = data.drop(columns=['total price', 'fire insurance'])



from sklearn.model_selection import train_test_split



train, test = train_test_split(data, test_size=0.2)
corr = train.corr().abs()

corr.RentAmountLog[corr.RentAmountLog >= 0.5].sort_values(ascending=False)
plt.subplot2grid((2,1), (0,0))

train['rent amount'].plot(kind='kde')

plt.title('Rent Amount')



plt.subplot2grid((2,1),(1,0))

train['RentAmountLog'].plot(kind='kde')

plt.title('Rent Amount Log')



plt.subplots_adjust(hspace=0.5, wspace = 0.3)

plt.show()





y_train = train['RentAmountLog']

y_test = test['RentAmountLog']

x_train = train.drop(columns=['RentAmountLog', 'rent amount'])

x_test = test.drop(columns=['RentAmountLog', 'rent amount'])
import xgboost

from sklearn.metrics import mean_squared_error



xgb = xgboost.XGBRegressor(min_child_weight=1.8, silent=1, colsample_bytree=0.8, subsample=1,

                           learning_rate=0.01, max_depth=4, n_estimators=3000,

                           reg_lambda=0.1, gamma = 0, silence=False, reg_alpha=0.1, nthr = -1 )



xgb.fit(x_train, y_train)

y_test_pred = xgb.predict(x_test)

RMSE = np.sqrt(mean_squared_error(y_test_pred, y_test))

print(RMSE.round(6))