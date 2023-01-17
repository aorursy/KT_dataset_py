import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

from fbprophet import Prophet

from statsmodels.tsa.statespace.varmax import VARMAX

import warnings

warnings.filterwarnings('ignore')
train_orgnl = pd.read_csv('../input/into-the-future/train.csv')

test = pd.read_csv('../input/into-the-future/test.csv')
train_orgnl
train_orgnl.info()
train_orgnl['time'] = pd.to_datetime(train_orgnl.time)

test['time'] = pd.to_datetime(test.time)
test.info()
print(train_orgnl['time'].min())

print(train_orgnl['time'].max())
train = train_orgnl.copy()
train_orgnl.set_index('time', inplace=True)

train_orgnl.drop('id', axis=1, inplace=True)
train_orgnl['feature_1'].plot(kind='line',color='red', figsize=(20,10))
train_orgnl['feature_2'].plot(kind='line',color='red', figsize=(20,10))
print('Number of missing values in feature_1 of Training Data: ' ,train['feature_1'].isnull().sum())

print('Number of missing values in feature_2 of Training Data: ' ,train['feature_2'].isnull().sum())

print('Number of missing values in feature_1 of Test Data: ' ,test['feature_1'].isnull().sum())
train1 = train[:int(0.9*len(train))]

valid = train[int(0.9*len(train)):]
print(train1.shape)

print(valid.shape)
mdl = VARMAX(train_orgnl)

mdl_fit = mdl.fit()
prdn = mdl_fit.forecast(steps=len(valid))

prdn.head(10)
valid.shape
from sklearn.metrics import mean_squared_error as ms

from sklearn.metrics import mean_absolute_error as ma
valid.set_index('time', inplace=True)

valid.drop('id',axis=1, inplace=True)
prdn.head()
import math

rmse=math.sqrt(ms(prdn,valid))

print('Mean absolute error is: '+ str(ma(prdn,valid)))

print('Root Mean Squared error is: ' + str(rmse))
fbp_data = pd.DataFrame(columns=['ds', 'y', 'add1'])

fbp_data['ds'] = train['time']

fbp_data['y'] = train['feature_2']

fbp_data['add1'] = train['feature_1']
size = int(fbp_data.shape[0]*0.9)

x_train = fbp_data[:size]

x_valid = fbp_data[size:]
model = Prophet()

model.fit(x_train)
print(x_train.shape)

print(x_valid.shape)
pred = model.predict(x_valid.drop('y', axis=1))
pred
model.plot_components(pred)
f_prediction = pred['yhat']
f_prediction.head()
f2_valid = x_valid['y']

plt.plot(f_prediction, 'r')

plt.plot(f2_valid.reset_index(drop=True), 'b')
rmse_fb=np.sqrt(ms(f2_valid, f_prediction))

print('Mean absolute error is: '+ str(ma(f_prediction,f2_valid)))

print('Root Mean Squared error is: ' + str(rmse_fb))
model_full = Prophet()

model_full.fit(fbp_data)
id_test = test['id'].values

date_test = test['time'].values
x_test = pd.DataFrame(data=date_test, columns=['ds'])

x_test['add1'] = test['feature_1'].values
prediction_test = model.predict(x_test)
model_full.plot_components(prediction_test)
prediction = prediction_test['yhat']
prediction_test.head()
prediction_test.to_csv('results.csv')