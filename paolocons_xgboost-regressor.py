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
data = pd.read_csv('../input/another-fiat-500-dataset-1538-rows/automobile_dot_it_used_fiat_500_in_Italy_dataset_filtered.csv')

data.head()
data.engine_power.unique()
import matplotlib.pyplot as plt

import seaborn as sns



numerical_features = data.select_dtypes(exclude=['object']).drop(['price'], axis=1).copy()

print(numerical_features.columns)



fig = plt.figure(figsize=(12,18))

for i in range(len(numerical_features.columns)):

    fig.add_subplot(9,4,i+1)

    sns.boxplot(y=numerical_features.iloc[:,i])



plt.tight_layout()

plt.show()



fig = plt.figure(figsize=(12,18))

for i in range(len(numerical_features.columns)):

    fig.add_subplot(9, 4, i+1)

    sns.scatterplot(numerical_features.iloc[:, i],data['price'])

plt.tight_layout()

plt.show()
figure, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2)

figure.set_size_inches(16,28)

_ = sns.regplot(data['engine_power'], data['price'], ax=ax1)

_ = sns.regplot(data['age_in_days'], data['price'], ax=ax2)

_ = sns.regplot(data['km'], data['price'], ax=ax3)

_ = sns.regplot(data['previous_owners'], data['price'], ax=ax4)

_ = sns.regplot(data['lat'], data['price'], ax=ax5)

_ = sns.regplot(data['lon'], data['price'], ax=ax6)

data = data.drop(data[(data['engine_power']>55) & (data['engine_power']<60)].index)

data = data.drop(data[(data['engine_power']>62) & (data['engine_power']<70)].index)

data = data.drop(data[(data['engine_power']>75)].index)

# it turned out that removing spurious values on this feature improved MAE from 553 to 503

data = data.drop(data[(data['age_in_days']<2800) & (data['price']<5000)].index)

data = data.drop(data[(data['age_in_days']>2500) & (data['price']>7900)].index)

data = data.drop(data[(data['age_in_days']>3500) & (data['price']>8000)].index)

data = data.drop(data[(data['age_in_days']>4000) & (data['price']<3000)].index)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor



data2 = data.copy()

data2 = pd.get_dummies(data)



feature_to_drop = ['price','previous_owners']



X = data2.drop(feature_to_drop, axis=1)

Y = data2.price



len(X), len(Y)



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state=0)



#scale

mms = MinMaxScaler()



mms.fit_transform(X_train)

mms.transform(X_test)





# Define the model

xgbr = XGBRegressor(n_estimators=300, random_state=0)



# Fit the model

xgbr.fit(X_train,Y_train)



# Get predictions

Y_pred_train = xgbr.predict(X_train)

Y_pred_test = xgbr.predict(X_test)



# Calculate MAE

mae_train = mean_absolute_error(Y_train,Y_pred_train)

print("Mean Absolute Error:" , mae_train)

mae_test = mean_absolute_error(Y_test,Y_pred_test)

print("Mean Absolute Error:" , mae_test)
# plot the difference scatter vs the target

import matplotlib.pyplot as plt



fig = plt.figure()

ax = fig.add_subplot(111)

plt.scatter(Y_pred_test, Y_test - Y_pred_test)

plt.grid(True)

plt.xlabel('Target')

plt.ylabel('Difference in prediction')

plt.show()