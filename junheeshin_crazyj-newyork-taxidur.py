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
inputdir = '../input/'

outputdir = './'

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb

from sklearn.model_selection import train_test_split

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
dftrain = pd.read_csv(inputdir+'train.csv')

dftest = pd.read_csv(inputdir+'test.csv')

dftrain.store_and_fwd_flag.replace(['Y','N'], [1, 0], inplace=True)

dftest.store_and_fwd_flag.replace(['Y','N'], [1, 0], inplace=True)
dftest.columns
# get day of week. 

# get time of day.

picktime = pd.to_datetime(dftrain['pickup_datetime']).dt

dftrain['pickup_dayofweek'] = picktime.dayofweek

dftrain['pickup_timeofday'] = picktime.hour*60+picktime.minute



picktime = pd.to_datetime(dftest['pickup_datetime']).dt

dftest['pickup_dayofweek'] = picktime.dayofweek

dftest['pickup_timeofday'] = picktime.hour*60+picktime.minute
# 요일별 이동시간 평균값

dftrain[['pickup_dayofweek', 'trip_duration']].groupby(['pickup_dayofweek']).mean().plot.bar()

# dftrain.head()



# 요일별 카운팅. monday=0

sns.countplot('pickup_dayofweek', data=dftrain)

# 월요일이 가장 손님이 적고, 계속 올라가서 금요일이 가장 많다. 
dftrain['log_trip_duration'] = np.log(dftrain['trip_duration']+1)
plt.hist( dftrain['log_trip_duration'].values, bins=100)

plt.xlabel('log trip dur')

plt.ylabel('number of rec')

plt.show()
dftrain['log_trip_duration'].describe()

dftrain.describe()
N=10000

city_long_border = (-75, -75)

city_lat_border = (40, 40)

fig, ax = plt.subplots(1,2, sharex=True, sharey=True)

ax[0].scatter( dftrain['pickup_longitude'].values[:N],

             dftrain['pickup_latitude'].values[:N], color='blue', s=1, label='train', alpha=0.1)

ax[1].scatter( dftest['pickup_longitude'].values[:N],

             dftest['pickup_latitude'].values[:N],

             color='green', s=1, label='train', alpha=0.1)

plt.show()
# train model

feature_names = list(['pickup_dayofweek', 'pickup_timeofday', 'passenger_count',

                     'pickup_longitude', 'pickup_latitude', 'store_and_fwd_flag'])

y = np.log(dftrain['trip_duration'].values+1)

x_train = dftrain[feature_names].values

datamean = x_train.mean(axis=0)

datastd = x_train.std(axis=0)



x_train, x_val, y_train, y_val = train_test_split((dftrain[feature_names].values-datamean)/datastd, 

                                                  y, test_size=0.1)

x_test = (dftest[feature_names].values - datamean)/datastd
print(datamean, datastd)

# x_train = x_train[:N]

# y_train = y_train[:N]
dfy = pd.DataFrame(y_train)

dfy.describe()
xgtrain = xgb.DMatrix(x_train, label=y_train)

xgval = xgb.DMatrix(x_val, label=y_val)

xgtest = xgb.DMatrix(x_test)



watchlist = [(xgtrain, 'train'), (xgval, 'valid')]

xgb_pars = {'min_child_weight': 50, 'eta': 0.1, 'colsample_bytree': 0.5, 'max_depth': 10,

            'subsample': 0.8, 'lambda': 1., 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,

            'eval_metric': 'rmse', 'objective': 'reg:linear'}
model=None
tmpFile='model.h5'

if not os.path.exists(tmpFile):

    model = xgb.train(xgb_pars, xgtrain, 400, watchlist, early_stopping_rounds=50,

                      maximize=False, verbose_eval=10)

else:

    print("continuous")

    model = xgb.train(xgb_pars, xgtrain, 400, watchlist, early_stopping_rounds=50,

                      maximize=False, verbose_eval=10, xgb_model=tmpFile)

model.save_model(tmpFile)
ypred = model.predict(xgval)

fig,ax = plt.subplots(ncols=2)

ax[0].scatter(ypred, y_val, s=0.1, alpha=0.1)

ax[0].set_xlabel('log(prediction)')

ax[0].set_ylabel('log(ground truth)')

ax[0].set_xlim([0,10])

ax[0].set_ylim([0,10])

ax[1].scatter(np.exp(ypred), np.exp(y_val), s=0.1, alpha=0.1)

ax[1].set_xlabel('prediction')

ax[1].set_ylabel('ground truth')

ax[1].set_xlim([0,3000])

ax[1].set_ylim([0,3000])

plt.show()
cor_y = pd.DataFrame({'ypred':ypred, 'yval':y_val})

cor_y.corr().style.background_gradient(cmap='coolwarm')

ytest = model.predict(xgtest)

predr = np.exp(ytest)

submission = pd.read_csv(inputdir+'sample_submission.csv')

submission["trip_duration"] = predr.astype('int')

submission.to_csv("submission.csv", index=False)

submission.head()


if False:

    import keras

    from keras.models import Sequential

    from keras.layers import Dense, Dropout

    from keras.callbacks import EarlyStopping



    model = Sequential()

    model.add( Dense(64, input_dim = x_train.shape[1], init='he_normal',

                     activation='relu'))

    # model.add( Dense(256, init='he_normal', activation='relu'))

    # model.add( Dense(128, init='he_normal', activation='relu'))

    model.add( Dense(32, init='he_normal', activation='relu'))

    model.add( Dense(32, init='he_normal', activation='relu'))

    model.add( Dense(32, init='he_normal', activation='relu'))

    model.add( Dense(32, init='he_normal'))

    model.add( Dense(1, init='uniform'))

    model.summary()





    model.compile(loss='mse', optimizer='rmsprop')



    es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')



    history = model.fit(x_train, y_train, batch_size=128, epochs=200, 

                       validation_data=(x_val, y_val), callbacks=[es])



    pred = model.predict(x_test)

    predr = np.exp(pred)

    print('pred=', pred)

    # print('y_val=', y_val)

    dfpred = pd.DataFrame(pred)

    dfpred.plot.hist(bins=20)



    dftrain['log_trip_duration'].plot.hist(bins=20)



    #submission

    submission = pd.read_csv(inputdir+'sample_submission.csv')

    submission.head()



    submission["trip_duration"] = predr.astype('int')

    submission.to_csv("submission.csv", index=False)

    submission.head()