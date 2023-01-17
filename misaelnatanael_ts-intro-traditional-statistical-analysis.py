import numpy as np

import pandas as pd

from pandas import Series, DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

df.date = pd.to_datetime(df.date, format='%d.%m.%Y')

df
df.describe()
print('Number of records with negative item_price: %d' %(len(df[df['item_price'] < 0])))

print('Number of records with negative item_cnt_day: %d' %(len(df[df['item_cnt_day'] < 0])))
df[df['item_price']<0]
df[df['item_id'] == 2973].item_price.unique()
df = df[df['item_price'] > 0]
df[df['item_cnt_day']<0]
print('Proportion of records with negative item_cnt_day: %f%%' %(100*len(df[df['item_cnt_day']<0])/len(df)))
df.loc[df['item_cnt_day']<0,'item_cnt_day'] = 0
df.describe()
f, (ax1, ax2) = plt.subplots(figsize=(14,4), nrows=1, ncols=2)

sns.boxplot(df['item_price'].dropna(), ax=ax1)

sns.boxplot(df['item_cnt_day'].dropna(), ax=ax2)

f.suptitle('Distribution of item price and item sales per day')
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

items[items['item_id']==df[df['item_price']==df['item_price'].max()].item_id.values[0]]  
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

items[items['item_id']==df[df['item_cnt_day']==df['item_cnt_day'].max()].item_id.values[0]]  
per_shop = df.groupby(['shop_id','date'],as_index=False)['item_cnt_day'].sum()

per_shop
for c in range(0,20):

    fig = plt.figure(figsize=(12,3))

    plt.plot(per_shop[per_shop.shop_id==c].set_index('date')['item_cnt_day'])

    plt.title('shop_id: %d' %(c))
daily = df.groupby(['date'],as_index=False)['item_cnt_day'].sum()

daily = daily.set_index('date')

plt.figure(figsize=(10,6))

sns.lineplot(data=daily)

plt.title('Total sales per day from January 2013 - October 2015')
plt.figure(figsize=(10,6))

sns.lineplot(data=daily.set_index(daily.index.strftime('%d')).head(30))

plt.title('Total sales per day in January 2013')
daily['weekday'] = daily.index.strftime('%A')



plt.figure(figsize=(15,6))

sns.lineplot(x='date', y='item_cnt_day', hue='weekday', hue_order = daily.iloc[5:].weekday.unique(), 

             data=daily.reset_index().iloc[0*365:1*365,:])

plt.title('Total sales per day in 2013')

plt.figure(figsize=(15,6))

sns.lineplot(x='date', y='item_cnt_day', hue='weekday', hue_order = daily.iloc[5:].weekday.unique(), 

             data=daily.reset_index().iloc[1*365:2*365,:])

plt.title('Total sales per day in 2014')

plt.figure(figsize=(15,6))

sns.lineplot(x='date', y='item_cnt_day', hue='weekday', hue_order = daily.iloc[5:].weekday.unique(), 

             data=daily.reset_index().iloc[2*365:3*365,:])

plt.title('Total sales per day in 2015')
df=df.set_index('date')

cols = ['date_block_num','shop_id','item_id','item_price']
#weekly sales

weekly = df.resample('w').sum().drop(columns=cols)

plt.figure(figsize=(10,6))

sns.lineplot(data=weekly)

plt.title('Total sales per week from January 2013 - October 2015')



#monthly sales

monthly = df.resample('M').sum().drop(columns=cols)

plt.figure(figsize=(10,6))

sns.lineplot(data=monthly)

plt.title('Total sales per month from January 2013 - October 2015')
from statsmodels.tsa.seasonal import seasonal_decompose

fig, ax = plt.subplots(4,2,figsize=(15,4))



#weekly sales

decomp1 = seasonal_decompose(weekly,model='additive')

decomp1.observed.plot(ax=ax[0,0],ylabel='Observed',legend=None)

decomp1.trend.plot(ax=ax[1,0],ylabel='Trend',legend=None,sharex=ax[0,0])

decomp1.seasonal.plot(ax=ax[2,0],ylabel='Seasonal',legend=None,sharex=ax[0,0])

decomp1.resid.plot(ax=ax[3,0],ylabel='Resid',legend=None,sharex=ax[0,0])

ax[3,0].set_title('Weekly sales additive decomposition',x=0.5,y=4.8)



#monthly sales

decomp2 = seasonal_decompose(monthly,model='additive')

decomp2.observed.plot(ax=ax[0,1],ylabel='Observed',legend=None)

decomp2.trend.plot(ax=ax[1,1],ylabel='Trend',legend=None,sharex=ax[0,0])

decomp2.seasonal.plot(ax=ax[2,1],ylabel='Seasonal',legend=None,sharex=ax[0,0])

decomp2.resid.plot(ax=ax[3,1],ylabel='Resid',legend=None,sharex=ax[0,0])

ax[3,1].set_title('Monthly sales additive decomposition',x=0.5,y=4.8)
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import kpss

ts = Series(daily.dropna().item_cnt_day)

result1 = adfuller(ts)

print('-----ADF test-----')

print('ADF Statistic: %f' % result1[0])

print('p-value: %f' % result1[1])

print('Critical Values:')

for key, value in result1[4].items():

	print('\t%s: %.3f' % (key, value))

print('\n')

    

print('-----KPSS test-----')

result2 = kpss(ts,lags='auto')

print('KPSS Statistic: %f' % result2[0])

print('p-value: %f' % result2[1])

print('Critical Values:')

for key, value in result2[3].items():

	print('\t%s: %.3f' % (key, value))
plt.figure(figsize=(9,6))

sns.lineplot(data=daily.item_cnt_day.diff(1))

plt.xlabel('date')

plt.ylabel('differenced total sales')
ts = Series(daily.item_cnt_day.diff(1).dropna())

result1 = adfuller(ts)

print('-----ADF test-----')

print('ADF Statistic: %f' % result1[0])

print('p-value: %f' % result1[1])

print('Critical Values:')

for key, value in result1[4].items():

	print('\t%s: %.3f' % (key, value))

print('\n')

    

print('-----KPSS test-----')

result2 = kpss(ts,lags='auto')

print('KPSS Statistic: %f' % result2[0])

print('p-value: %f' % result2[1])

print('Critical Values:')

for key, value in result2[3].items():

	print('\t%s: %.3f' % (key, value))
from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf

ts = Series(daily.dropna().item_cnt_day)

fig, ax = plt.subplots(1,2,figsize=(15,4))

plot_acf(ts,ax=ax[0],lags=45)

plot_pacf(ts,ax=ax[1],lags=45)

plt.show()
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_absolute_error as MAE

warnings.filterwarnings("ignore")



#split training and test set

ts = Series(daily.dropna().item_cnt_day)

train_size = int(len(ts) * 0.8)

train, test = ts[0:train_size], ts[train_size:]

history = [x for x in train]



#make predictions with expanding window method

preds = list()

for t in range(len(test)):

    model = SARIMAX(history, order=(1,1,3),seasonal_order=(1,0,1,7),enforce_stationarity=False, 

                    enforce_invertibility=False).fit()

    yhat = model.forecast()[0]

    preds.append(yhat)

    history.append(test[t])

    

# calculate out of sample error

score = MAE(test, preds)

print('Mean absolute error(MAE) score: %f' %(score))
res = model.resid

fig,ax = plt.subplots(2,1,figsize=(15,8))

fig = plot_acf(res, ax=ax[0])

fig = plot_pacf(res, ax=ax[1])

plt.suptitle('Correlation plot for residuals')

plt.show()
predicted = Series(preds,index=test.index)

plt.figure(figsize=(10,6))

sns.lineplot(data=ts)

sns.lineplot(data=predicted,color='orange')

plt.axvline(x=ts.index[train_size],color='red',linestyle='dashed')

plt.title('Forecasting with SARIMA(1,1,3)(1,0,1)7 model')
plt.figure(figsize=(10,6))

sns.lineplot(data=ts[train_size:])

sns.lineplot(data=predicted,color='orange')

plt.title('Forecasting result on the test period')