import pandas as pd

import seaborn as sns

import numpy as np

import datetime

from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing
train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')

test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')

data = train.append(test, sort=False)
data.head()
data['datetime'] = data['datetime'].astype('datetime64[ns]')

data['Day'] = pd.DatetimeIndex(data['datetime']).day

data['Month'] = pd.DatetimeIndex(data['datetime']).month

data['Year'] = pd.DatetimeIndex(data['datetime']).year

data['Hour'] = pd.DatetimeIndex(data['datetime']).hour

data['weekday'] = pd.DatetimeIndex(data['datetime']).weekday
df=data.drop(['registered','casual','atemp','Day','season'],axis=1)
df[df["windspeed"]==0].head()
%config InlineBackend.figure_format = 'retina'

corr1 = df[['temp','humidity','windspeed']].corr() 

mask = np.zeros_like(corr1)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr1, vmax=0.8, mask=mask, square=True,annot=True)
lb = preprocessing.LabelBinarizer()

df.Year = lb.fit_transform(df.Year)

cont=['temp','humidity','windspeed']

feat=df[cont]

minmax_scale = preprocessing.MinMaxScaler().fit(feat.values)

df[cont] = minmax_scale.transform(feat.values)
df.head()
df[df["windspeed"]==0].head()
categ=['holiday','weather','Month','Year','Hour','weekday','workingday']

for var in categ:

    df[var] = df[var].astype("category")

df = pd.get_dummies(data=df, columns=['holiday','weather','workingday','Year'])

df.head()
df_train = df[pd.notnull(df['count'])].sort_values(by=['datetime'])

df_test = df[~pd.notnull(df['count'])].sort_values(by=['datetime'])
df_train=df_train.drop(['datetime'],axis=1)

test=df_test.drop(['datetime','count'],axis=1)


df_train['count'].plot(kind="hist", bins=100)
df_train['count'] = np.log1p(df_train['count'])
X = df_train.drop(['count'],axis=1)

y=df_train['count']

X.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.80, random_state=1)

rfr = RandomForestRegressor(n_estimators = 100)

rfr.fit(X_train, y_train)

pred = rfr.predict(X_test)

sns.scatterplot(x = y_test, y = pred)
from sklearn import metrics

print('RMSLE:', np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(pred))))
from sklearn.ensemble import GradientBoostingRegressor

gbm = GradientBoostingRegressor(n_estimators=2000,alpha=0.01)

gbm.fit(X_train,y_train)

preds = gbm.predict(X_test)

print('RMSLE:', np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(preds))))
algo_gbm = gbm.predict(X_test)

algo_rf = rfr.predict(X_test)

algo_mean =np.expm1(algo_gbm)*0.9 + np.expm1(algo_rf)*0.1

print('RMSLE:', np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), algo_mean)))
algo_gbm_tst = gbm.predict(test)

algo_rf_tst = rfr.predict(test)

algo_mean_tst =np.expm1(algo_gbm_tst)*0.9 + np.expm1(algo_rf_tst)*0.1
submission = pd.DataFrame({'datetime':df_test['datetime'],'count':algo_mean_tst})

submission.head()
submission.to_csv('Submission.csv',index=False)