import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from fbprophet import Prophet

from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.style.use('fivethirtyeight')
np.random.seed(42)



train = pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv',

                    parse_dates=['week_start_date'])

test = pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv',

                   parse_dates=['week_start_date'])

train['cases'] = pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv', usecols=[3])
#NULL ANALYSIS

if train.isnull().sum().any():

    null_cnt = train.isnull().sum().sort_values()

    print('TRAIN null count:', null_cnt[null_cnt > 0])



if test.isnull().sum().any():

    null_cnt = test.isnull().sum().sort_values()

    print('TEST null count:', null_cnt[null_cnt > 0])
train.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)

test.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
train.isnull().sum().any(),test.isnull().sum().any()
ltrain = train.shape[0]

df = pd.concat([train,test], sort=False).set_index('week_start_date')  

print('Combined df shape:{}'.format(df.shape))
# drop constant column

constant_column = [col for col in df.columns if df[col].nunique() == 1]

print('drop CONSTANT columns:', constant_column)

df.drop(constant_column, axis=1, inplace=True)



corr_matrix = df.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [c for c in upper.columns if any(upper[c] > 0.98)]

del upper



print('drop SIMILAR columns:', to_drop)

df.drop(to_drop,1,inplace=True)
train = df[:ltrain].copy()

test = df[ltrain:].copy()

del df
drop_cols = ['city']



y_sj = train[train.city=='sj']['cases']

y_iq = train[train.city=='iq']['cases']



train_sj = train[train.city=='sj'].drop(drop_cols,1)

train_iq = train[train.city=='iq'].drop(drop_cols,1)



test_sj = test[test.city=='sj'].drop(drop_cols,1)

test_iq = test[test.city=='iq'].drop(drop_cols,1)

model_sj = Prophet(yearly_seasonality=True,

                   weekly_seasonality = 1,

                   daily_seasonality=False,

                   seasonality_mode='multiplicative'

                ).add_seasonality(name='monthly', period=30.5, fourier_order=4, prior_scale=0.5

                ).add_seasonality(name='quarterly', period=365.25/4, fourier_order=12, prior_scale=1)
model_sj.fit(train_sj.reset_index().rename(columns={'week_start_date':'ds', 'cases':'y'}))

pred_sj = model_sj.predict(df=test_sj.reset_index().rename(columns={'week_start_date':'ds'}))
# Plot the forecast

f, ax = plt.subplots(1)

f.set_figheight(5)

f.set_figwidth(15)

fig = model_sj.plot(pred_sj, ax=ax)
# Plot the components

fig = model_sj.plot_components(pred_sj)
model_iq = Prophet(yearly_seasonality=True,

                    weekly_seasonality=1,

                    daily_seasonality=False,

                    seasonality_mode='multiplicative'

                ).add_seasonality(name='monthly', period=30.5, fourier_order=4

                )

model_iq.fit(train_iq.reset_index().rename(columns={'week_start_date':'ds', 'cases':'y'}))

pred_iq = model_iq.predict(df=test_iq.reset_index().rename(columns={'week_start_date':'ds'}))
# Plot the forecast

f, ax = plt.subplots(1)

f.set_figheight(5)

f.set_figwidth(15)

fig = model_iq.plot(pred_iq, ax=ax)
preds = np.concatenate((pred_sj.yhat.values, pred_iq.yhat.values), axis=0).clip(0)
test['total_cases'] = np.round(preds,0)

test[['city','year','weekofyear','total_cases']].to_csv('submission_prophet.csv', float_format='%.0f', index=False)