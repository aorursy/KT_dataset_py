# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.preprocessing import RobustScaler

from fbprophet import Prophet

from sklearn.metrics import mean_squared_error, mean_absolute_error

import math

from statsmodels.tsa.seasonal import seasonal_decompose
%matplotlib inline

df_train = pd.read_csv('../input/into-the-future/train.csv')

df_test = pd.read_csv('../input/into-the-future/test.csv')
df_train.head()
df_train.info()
df_train.describe()
df_test.info()
df_test.describe()
# Saving id of test

id_test = df_test['id'].values

date_test = df_test['time'].values
# Train time stamp

df_train['time'] = pd.to_datetime(df_train['time'])

df_train.info()
df_train.head()
# Train time stamp

df_test['time'] = pd.to_datetime(df_test['time'])

df_test.info()
feature_1_total = df_train['feature_1']

feature_1_total = feature_1_total.append(pd.Series(df_test['feature_1'].values))

print('Total entries in train and test data combined is :',len(feature_1_total.values))
feature_1_total.describe()
# Plotting feature_1 of combined train and test data

plt.plot(feature_1_total.values)
# Plotting feature_1 of train data.

plt.plot(df_train['feature_1'])
# Plotting feature_1 of test data

plt.plot(df_test['feature_1'])
# Zoomed plot to check seasonality in data

plt.plot((df_train.loc[100:200, 'feature_2']), 'r')
df_train_time_indexed = df_train['feature_2']

df_train_time_indexed.index = df_train['time']
# Correlation matrix

df_train.corr()
sns.heatmap(df_train.corr())
# Dropping ID

df_train = df_train.drop('id', axis=1)
# pair plot to understand kind of relationship between various features

sns.pairplot(data=df_train)
# Checking distribution of the feature_1 and feature_2

fig, ax1 = plt.subplots(ncols=2, figsize=(20,5))

ax1[0].hist(df_train['feature_1'])

ax1[0].set_title('Feature_1 distribution')

ax1[1].hist(df_train['feature_2'])

ax1[1].set_title('Feature_2 distribution')
# Creating dataframe for FBProphet

X = pd.DataFrame(columns=['ds', 'y', 'add1'])

X['ds'] = df_train['time']

X['y'] = df_train['feature_2']

X['add1'] = df_train['feature_1']
X.describe()
# Checking relationship between feature_1 and feature_2

sns.relplot(x='add1', y='y', data=X)
# Splitting data to evaluate model

size = int(X.shape[0]*0.9)

x_train = X[:size]

x_valid = X[size:]
print('Train data size = {}, Valid data size ={}'.format(x_train.shape[0], x_valid.shape[0]))
# y_true_valid

feature_2_valid = x_valid['y']
# Wanted to fit logistic trend, seasonality but we don't have enough data knowledge

model = Prophet(growth='linear', n_changepoints=50, seasonality_mode='multiplicative')

model.fit(x_train)
#Forecasting based on x_valid

forecast = model.predict(x_valid.drop('y', axis=1))
model.plot_components(forecast)
# Saving Prediction in a variable

prediction = forecast['yhat']
plt.plot(prediction, 'r')

plt.plot(feature_2_valid.reset_index(drop=True), 'b')
print('RMSE :', math.sqrt(mean_squared_error(feature_2_valid, prediction)))
print('MAE :', mean_absolute_error(feature_2_valid, prediction))

X_test = pd.DataFrame(data=date_test, columns=['ds'])

X_test['add1'] = df_test['feature_1'].values
X_test.head()
model_full = Prophet(growth='linear', n_changepoints=50, seasonality_mode='multiplicative')

model_full.fit(X)

forecast_test = model.predict(X_test)
model_full.plot_components(forecast_test)
prediction = forecast_test['yhat']
forecast_test.head()
submisson = pd.DataFrame({'id': id_test,

                         'feature_2': prediction})
submisson.head()
submisson.to_csv(r'Final_Submit.csv', index=False)