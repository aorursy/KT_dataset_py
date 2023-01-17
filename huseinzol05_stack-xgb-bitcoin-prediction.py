import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import time

from datetime import datetime

from datetime import timedelta

from sklearn import cross_validation

from sklearn.ensemble import *

sns.set()
df = pd.read_csv('../input/all-crypto-currencies/crypto-markets.csv', parse_dates=['date'], index_col='date')

df = df[df['symbol']=='BTC']

df.drop(['volume','symbol','name','ranknow','market'],axis=1,inplace=True)

df.head()
df['close'].plot(figsize=(12,6),label='Close')

df['close'].rolling(window=30).mean().plot(label='30 Day Avg')

plt.legend()
period = 30

minmax = MinMaxScaler().fit(df.iloc[:, 3].values.reshape((-1,1)))

close_normalize = minmax.transform(df.iloc[:, 3].values.reshape((-1,1)))

normalized = pd.DataFrame(close_normalize)

normalized['Price_After_period']=normalized[0].shift(-period)

normalized.dropna(inplace=True)

X=normalized.drop('Price_After_period',axis=1)

print(normalized.head())

y=normalized['Price_After_period']

print(X.head())

y.head()
train_X,test_X,train_Y,test_Y=cross_validation.train_test_split(X,

                                                                y,

                                                                test_size=0.2,random_state=101)
from sklearn.ensemble import *

ada = AdaBoostRegressor(n_estimators=500, learning_rate=0.1)

bagging = BaggingRegressor(n_estimators=500)

et = ExtraTreesRegressor(n_estimators=500)

gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1)

rf = RandomForestRegressor(n_estimators=500,random_state=101)
ada.fit(train_X, train_Y)

bagging.fit(train_X, train_Y)

et.fit(train_X, train_Y)

gb.fit(train_X, train_Y)

rf.fit(train_X, train_Y)
accuracy=ada.score(test_X, test_Y)

accuracy=accuracy*100

accuracy = float("{0:.4f}".format(accuracy))

print('Adaptive Accuracy is:',accuracy,'%')
accuracy=bagging.score(test_X, test_Y)

accuracy=accuracy*100

accuracy = float("{0:.4f}".format(accuracy))

print('Bagging Accuracy is:',accuracy,'%')
accuracy=et.score(test_X, test_Y)

accuracy=accuracy*100

accuracy = float("{0:.4f}".format(accuracy))

print('Extra Trees Accuracy is:',accuracy,'%')
accuracy=gb.score(test_X, test_Y)

accuracy=accuracy*100

accuracy = float("{0:.4f}".format(accuracy))

print('Gradient Boosting Accuracy is:',accuracy,'%')
accuracy=rf.score(test_X, test_Y)

accuracy=accuracy*100

accuracy = float("{0:.4f}".format(accuracy))

print('Random Forest Accuracy is:',accuracy,'%')
ada_out = ada.predict(test_X)

bagging_out = bagging.predict(test_X)

et_out = et.predict(test_X)

gb_out = gb.predict(test_X)

rf_out = rf.predict(test_X)

stack_predict = np.vstack([ada_out,bagging_out,et_out,gb_out,rf_out,test_Y]).T

corr_df = pd.DataFrame(stack_predict, columns=['ada','bagging','et','gb','rf','test'])

plt.figure(figsize=(10,5))

sns.heatmap(corr_df.corr(), annot=True)

plt.show()
corr_df.head()
corr_df.ada = minmax.inverse_transform(corr_df.ada.values.reshape((-1,1))).flatten()

corr_df.bagging = minmax.inverse_transform(corr_df.bagging.values.reshape((-1,1))).flatten()

corr_df.et = minmax.inverse_transform(corr_df.et.values.reshape((-1,1))).flatten()

corr_df.gb = minmax.inverse_transform(corr_df.gb.values.reshape((-1,1))).flatten()

corr_df.rf = minmax.inverse_transform(corr_df.rf.values.reshape((-1,1))).flatten()

corr_df.test = minmax.inverse_transform(corr_df.test.values.reshape((-1,1))).flatten()
corr_df.head()
import xgboost as xgb



params_xgd = {

    'max_depth': 7,

    'objective': 'reg:linear',

    'learning_rate': 0.033,

    'n_estimators': 10000

    }

clf = xgb.XGBRegressor(**params_xgd)

stack_train = np.vstack([ada.predict(train_X),

                           bagging.predict(train_X),

                           et.predict(train_X),

                           gb.predict(train_X),

                          rf.predict(train_X)]).T



stack_test = np.vstack([ada.predict(test_X),

                           bagging.predict(test_X),

                           et.predict(test_X),

                           gb.predict(test_X),

                          rf.predict(test_X)]).T



clf.fit(stack_train, train_Y, eval_set=[(stack_test, test_Y)], 

        eval_metric='rmse', early_stopping_rounds=20, verbose=True)
fig, ax = plt.subplots(figsize=(10,10))

xgb.plot_importance(clf, ax=ax)

plt.show()
out_ada=X[0].tolist() + ada.predict(X[-period-10:]).tolist()

out_bagging=X[0].tolist() + bagging.predict(X[-period-10:]).tolist()

out_et=X[0].tolist() + et.predict(X[-period-10:]).tolist()

out_gb=X[0].tolist() + gb.predict(X[-period-10:]).tolist()

out_rf=X[0].tolist() + rf.predict(X[-period-10:]).tolist()
out_xgb=X[0].tolist()+clf.predict(np.vstack([ada.predict(X[-period-10:]),

                           bagging.predict(X[-period-10:]),

                           et.predict(X[-period-10:]),

                           gb.predict(X[-period-10:]),

                          rf.predict(X[-period-10:])]).T).tolist()
last_date=pd.to_datetime(df.iloc[-1].name)

print(last_date)

modified_date = last_date + timedelta(days=1)

date=pd.date_range(modified_date,periods=period,freq='D')
out_ada = minmax.inverse_transform(np.array(out_ada).reshape((-1,1))).flatten()

out_bagging = minmax.inverse_transform(np.array(out_bagging).reshape((-1,1))).flatten()

out_et = minmax.inverse_transform(np.array(out_et).reshape((-1,1))).flatten()

out_gb = minmax.inverse_transform(np.array(out_gb).reshape((-1,1))).flatten()

out_rf = minmax.inverse_transform(np.array(out_rf).reshape((-1,1))).flatten()

out_xgb = minmax.inverse_transform(np.array(out_xgb).reshape((-1,1))).flatten()
date_ori=pd.to_datetime(df.index.date[:-period+10]).strftime(date_format='%Y-%m-%d').tolist()+pd.Series(date).dt.strftime(date_format='%Y-%m-%d').tolist()
len(date_ori)
len(out_ada)
fig = plt.figure(figsize = (15,10))

ax = plt.subplot(111)

x_range = np.arange(df.shape[0])

x_range_future = np.arange(len(out_ada))

ax.plot(x_range, df.close, label = 'true Close')

ax.plot(x_range_future, out_ada, label = 'ADA predict Close')

ax.plot(x_range_future, out_bagging, label = 'BAGGING predict Close')

ax.plot(x_range_future, out_et, label = 'ET predict Close')

ax.plot(x_range_future, out_gb, label = 'GB predict Close')

ax.plot(x_range_future, out_rf, label = 'RF predict Close')

ax.plot(x_range_future, out_xgb, label = 'STACK XGB predict Close')

box = ax.get_position()

ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

ax.legend(loc = 'upper center', bbox_to_anchor= (0.5, -0.05), fancybox = True, shadow = True, ncol = 5)

plt.title('overlap stock market')

plt.xticks(x_range_future[::180], date_ori[::180])

plt.show()
from PIL import Image

bitcoin_im = Image.open('../input/bitcoinpic/Bitcoin-Logo-640x480.png')
fig = plt.figure(figsize = (15,10))

ax = plt.subplot(111)

x_range = np.arange(df.shape[0])

x_range_future = np.arange(len(out_ada))

ax.plot(x_range, df.close, label = 'true Close')

ax.plot(x_range_future, out_ada, label = 'ADA predict Close')

ax.plot(x_range_future, out_bagging, label = 'BAGGING predict Close')

ax.plot(x_range_future, out_et, label = 'ET predict Close')

ax.plot(x_range_future, out_gb, label = 'GB predict Close')

ax.plot(x_range_future, out_rf, label = 'RF predict Close')

ax.plot(x_range_future, out_xgb, label = 'STACK XGB predict Close')

box = ax.get_position()

ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

ax.legend(loc = 'upper center', bbox_to_anchor= (0.5, -0.05), fancybox = True, shadow = True, ncol = 5)

plt.title('overlap stock market')

plt.xticks(x_range_future[::180], date_ori[::180])

fig.figimage(bitcoin_im, 100, 120, zorder=3,alpha=.5)

plt.show()
fig = plt.figure(figsize = (15,10))

ax = plt.subplot(111)

x_range = np.arange(df.shape[0])

x_range_future = np.arange(len(out_ada))

ax.plot(x_range[-30:], df.close[-30:], label = 'true Close')

ax.plot(x_range_future[-40:], out_ada[-40:], label = 'ADA predict Close')

ax.plot(x_range_future[-40:], out_bagging[-40:], label = 'BAGGING predict Close')

ax.plot(x_range_future[-40:], out_et[-40:], label = 'ET predict Close')

ax.plot(x_range_future[-40:], out_gb[-40:], label = 'GB predict Close')

ax.plot(x_range_future[-40:], out_rf[-40:], label = 'RF predict Close')

ax.plot(x_range_future[-40:], out_xgb[-40:], label = 'STACK XGB predict Close')

box = ax.get_position()

ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

ax.legend(loc = 'upper center', bbox_to_anchor= (0.5, -0.05), fancybox = True, shadow = True, ncol = 5)

plt.title('overlap stock market')

plt.xticks(x_range_future[-40:][::5], date_ori[-40:][::5])

fig.figimage(bitcoin_im, 100, 120, zorder=3,alpha=.5)

plt.show()