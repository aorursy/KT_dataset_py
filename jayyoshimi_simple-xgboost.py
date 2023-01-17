# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/regressioncsv/datasets_88705_204267_Real estate.csv', index_col = "No")

data.head(20)
data.info()
dates2013 = data[data['X1 transaction date']<= 2013]

sns.distplot(a = dates2013['X1 transaction date'], kde = False)

data['X1 transaction date'].describe()
for i in range(1,13):

    print(i/12)
names  = {'X1 transaction date':'transaction_date',

          'X2 house age': 'house_age',

          'X3 distance to the nearest MRT station': 'distance_MRT_station',

          'X4 number of convenience stores' : 'number_convenience_stores', 

          'X5 latitude': 'latitude', 

          'X6 longitude': 'longitude',

          'Y house price of unit area': 'sale_price'}



data.rename(columns = names, inplace = True)
data.transaction_date = round((data.transaction_date - 2012) *12)
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor



X = data.drop('sale_price', axis = 1)

y = data.sale_price

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state= 92387, test_size = .15)

xgbmodel = XGBRegressor()
xgbmodel.fit(X_train, y_train)
predictions = xgbmodel.predict(X_valid)

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
ns = []

for i in range(1,21):

    ns.append(i * 50)
#looking for a more optimized values for n_estimators

maes = []

for n in ns:

    model = XGBRegressor(n_estimators = n)

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    MAE = mean_absolute_error(predictions, y_valid)

    maes.append(MAE)

    print('For {} n_estimators:'.format(n))

    print('Mean Absolute Error = ' + str(MAE))

    print('========')
plt.figure(figsize = (6,3))

plt.plot(ns,maes)
learning_rates = [.5]

for i in range(0,20):

    learning_rates.append(learning_rates[-1]*0.8)

    

print(learning_rates)
maes = []

for l in learning_rates:

    model = XGBRegressor(n_estimators = 300, learning_rate = l)

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    MAE = mean_absolute_error(predictions, y_valid)

    maes.append(MAE)

    print('For n_estimators = {} and learning_rate = {}:'.format(300, l))

    print('Mean Absolute Error = ' + str(MAE))

    print('========')
plt.figure(figsize = (6,3))

plt.plot(learning_rates,maes)
maes = []



for lvalue in np.linspace(.001, .04, num = 20):

    model = XGBRegressor(n_estimators = 300, learning_rate = lvalue)

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    MAE = mean_absolute_error(predictions, y_valid)

    maes.append(MAE)

    print('For n_estimators = {} and learning_rate = {}:'.format(300, lvalue))

    print('Mean Absolute Error = ' + str(MAE))

    print('========')

    
plt.figure(figsize = (6,3))

plt.plot(np.linspace(.001, .04, num = 20),maes)
np.argmin(maes)

np.linspace(.001, .04, num = 20)[7]
LR = np.linspace(.001, .04, num = 20)[7]

ns = []

maes = []

for n in np.arange(50, 1000, step = 50):

    model = XGBRegressor(n_estimators = n, learning_rate = LR)

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    MAE = mean_absolute_error(predictions, y_valid)

    maes.append(MAE)

    print('For n_estimators = {} and learning_rate = {}:'.format(n, LR))

    print('Mean Absolute Error = ' + str(MAE))

    print('========')
plt.figure(figsize = (6,3))

plt.plot(np.arange(50, 1000, step = 50),maes)
rstates = [4539,3169,1003,2242]

n_est = 350 

lr = .01536842105263158



for r in rstates:

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state= r, test_size = .25)

    model = XGBRegressor(n_estimators = n_est, learning_rate = lr)

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    mae = mean_absolute_error(predictions, y_valid)

    print('MAE for this sampling: {}'.format(mae))