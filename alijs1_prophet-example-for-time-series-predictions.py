%matplotlib inline

import numpy as np

import pandas as pd

from fbprophet import Prophet

from sklearn.metrics import mean_squared_log_error
STORE = 1

ITEM = 105575

print('Reading data for store {}, item {}...'.format(STORE, ITEM))

it = pd.read_csv('../input/train.csv', iterator=True, chunksize=10000)

df = pd.concat([c[(c['store_nbr'] == STORE) & (c['item_nbr'] == ITEM)] for c in it])

print('Time-series data shape: {}'.format(df.shape))
TRAIN_SIZE = 365

CV_SIZE = 16 #if you make it bigger, fill missing dates in cv with 0 if any

X = df[-(TRAIN_SIZE+CV_SIZE):-CV_SIZE]

y = df[-CV_SIZE:]

print('Train on: {}, CV: {}'.format(X.shape, y.shape))



X = X[['date','unit_sales']]

X.columns = ['ds', 'y'] #Prophet names

print(X.tail())
m = Prophet(yearly_seasonality=True)

m.fit(X)

print('Prophet fitted')
future = m.make_future_dataframe(periods=CV_SIZE)

pred = m.predict(future)

print(pred[['ds','yhat','yhat_lower','yhat_upper']].tail(5))

m.plot(pred)
m.plot_components(pred)
pred['ds'] = pred['ds'].astype(str)

data = pred[['ds','yhat']].merge(y, left_on='ds', right_on='date')



items = pd.read_csv('../input/items.csv') #we need items for weights

items['weight'] = 1 + items['perishable'] * 0.25

data = data.merge(items[['item_nbr','weight']], how='left', on='item_nbr')



score = np.sqrt(mean_squared_log_error(data['unit_sales'].clip(0, 999999), data['yhat'].fillna(0).clip(0, 999999), data['weight']))

print('Score:{}'.format(score))