# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error

import seaborn as sns

import matplotlib.pyplot as plt

color = sns.color_palette()

sns.set_style('darkgrid')
X = pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent_v2.csv')

X.head()
X['total (R$)'].describe()
X.isnull().sum()
for i in X.columns:

    print(i, ':', X[i].dtype) 
X['floor'].head(20)
X['floor'].replace('-',np.NaN, inplace=True)

print("Number of null values - ",X['floor'].isnull().sum(),'\n')

X['floor'] = pd.to_numeric(X['floor'])

print(X['floor'].describe())
import math

X['floor'].replace(301,X['floor'].mean(), inplace=True)

X['floor'].fillna((math.floor(X['floor'].mean())), inplace=True)

print(X['floor'].describe())
X['floor'].head(20)
num_dtypes = X.select_dtypes(exclude='object')

numcorr = num_dtypes.corr()

plt.figure(figsize=(15,1))

sns.heatmap(numcorr.sort_values(by ='total (R$)',ascending=False).head(1),cmap='Blues')
plt.figure(figsize=(15,5))

sns.barplot(x = X['animal'], y=X['total (R$)'])
plt.figure(figsize=(15,5))

sns.barplot(x = X['city'], y=X['total (R$)'])
plt.figure(figsize=(15,5))

sns.barplot(x = X['furniture'], y=X['total (R$)'])
plt.figure(figsize=(15,5))

sns.scatterplot(X['area'],X['total (R$)'])
X = X.drop(X[(X['area']>10000) | ((X['total (R$)']>50000))].index)

plt.figure(figsize=(15,5))

sns.scatterplot(X['area'],X['total (R$)'])
plt.figure(figsize=(15,5))

sns.lineplot(x = X['fire insurance (R$)'],y=X['total (R$)'])
plt.figure(figsize=(15,5))

sns.scatterplot(x = X['hoa (R$)'],y=X['total (R$)'])

plt.figure(figsize=(15,5))

sns.barplot(x = X['parking spaces'], y=X['total (R$)'])
plt.figure(figsize=(15,5))

sns.barplot(x = X['bathroom'], y=X['total (R$)'])
plt.figure(figsize=(15,5))

sns.barplot(x = X['floor'], y=X['total (R$)'])
plt.figure(figsize=(15,5))

sns.barplot(x = X['rooms'], y=X['total (R$)'])
plt.figure(figsize=(15,5))

sns.lineplot(x = X['rent amount (R$)'], y=X['total (R$)'])
plt.figure(figsize=(15,5))

sns.scatterplot(x = X['property tax (R$)'], y=X['total (R$)'])
X['totprice'] = X['property tax (R$)']+X['rent amount (R$)']+ X['fire insurance (R$)']

plt.figure(figsize=(15,5))

sns.scatterplot(x = X['totprice'], y=X['total (R$)'])

X['totbhk'] = X['rooms'] + X['bathroom'] + X['parking spaces']

plt.figure(figsize=(15,5))

sns.barplot(x = X['totbhk'], y=X['total (R$)'])
y_data = X['total (R$)']

X_data = X.drop(['total (R$)','floor','property tax (R$)','rent amount (R$)','fire insurance (R$)','area','parking spaces','rooms','bathroom'], axis=1) 



X_train,X_valid,y_train,y_valid = train_test_split(X_data, y_data, train_size=0.8, test_size=0.2, random_state=0)
low_cardinality_cols = [cname for cname in X_train.columns if X_train[cname].nunique()<15 and X_train[cname].dtype=="object"]
X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)
my_model = XGBRegressor(n_estimators=2000, learning_rate=0.0099)

my_model.fit(X_train,y_train)
predictions = my_model.predict(X_valid)

mae = mean_absolute_error(predictions,y_valid)

print('Mean absolute error is', mae)
