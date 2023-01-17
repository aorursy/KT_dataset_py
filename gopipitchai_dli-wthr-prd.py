import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

%matplotlib inline



# from translate import Translator

from pandas import concat



from urllib.request import urlopen

from bs4 import BeautifulSoup



from math import sqrt

from numpy import concatenate

from matplotlib import pyplot

from pandas import read_csv

from pandas import DataFrame

from pandas import concat

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.models import model_from_json

from sklearn.externals import joblib
dt_comb = pd.read_csv("../input/wthr-cmbd/wthr_cmbd.csv")

dt_comb = dt_comb.iloc[:,1:]
dt_comb['dt'] = list(map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"), dt_comb.iloc[:,0]))
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]

    df = DataFrame(data)

    cols, names = list(), list()

    # input sequence (t-n, ... t-1)

    for i in range(n_in, 0, -1):

        cols.append(df.shift(i))

        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)

    for i in range(0, n_out):

        cols.append(df.shift(-i))

        if i == 0:

            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]

        else:

            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together

    agg = concat(cols, axis=1)

    agg.columns = names

    # drop rows with NaN values

    if dropnan:

        agg.dropna(inplace=True)

    return agg
dt_lst = []

for i in range(len(dt_comb)):

    dt_lst.append(dt_comb.loc[i,'dt'].date())

    

dt_comb['dt'] = dt_lst
lst = []

for var in ['baro', 'dew', 'temp']:

    dt_comb[var+'1'] = dt_comb[var]

    lst.append([var, var+'1'])
lst[0].append(5)

lst[1].append(7)

lst[2].append(4)
len(dt_comb)
RMSE = []



for var in lst:

    acc_df = pd.DataFrame()

    n_hours = 1

    n_features = 1

    no_of_days = var[-1]

    no_of_outcome = 1



    values = dt_comb.loc[~dt_comb[var[1]].isnull(), var[:-1]].values

    values = values.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled = scaler.fit_transform(values)

    

    scaler_filename = var[0] + "_scaler.save"

    joblib.dump(scaler, scaler_filename) 



    reframed = series_to_supervised(scaled[:,:-no_of_outcome], no_of_days, 1)

    

    if no_of_outcome > 1:

        reframed.drop(['var1(t)', 'var2(t)', 'var3(t)', 'var4(t)', 'var5(t)', 'var7(t)', 'var8(t)', 'var9(t)', 'var10(t)'], axis=1, inplace=True)



    # split into train and test sets

    values = reframed.values

    n_train_hours = len(values) - 100 - no_of_days

    train = values[:n_train_hours, :]

    test = values[n_train_hours:, :]

    # split into input and outputs

    train_X, train_y = train[:, :-no_of_outcome], train[:, -no_of_outcome]

    test_X, test_y = test[:, :-no_of_outcome], test[:, -no_of_outcome]

    # reshape input to be 3D [samples, timesteps, features]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    train_y = train_y.reshape((train_y.shape[0], 1))

    test_y = test_y.reshape((test_y.shape[0], 1))

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)



    # design network

    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))

    # model.add(Dropout(0.2))

    model.add(LSTM(50, return_sequences=True))

    # model.add(Dropout(0.2))

    model.add(LSTM(50, return_sequences=True))

    # model.add(Dropout(0.2))

    model.add(LSTM(50, return_sequences=True))

    # model.add(Dropout(0.2))

    model.add(LSTM(50, return_sequences=True))

    # model.add(Dropout(0.2))

    model.add(LSTM(50, return_sequences=True))

    # model.add(Dropout(0.2))

    model.add(LSTM(50, return_sequences=True))

    # model.add(Dropout(0.2))

    model.add(LSTM(50, return_sequences=True))

    # model.add(Dropout(0.2))

    model.add(LSTM(50, return_sequences=False))

    # model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(loss='mae', optimizer='adam')

    # fit network



    history = model.fit(train_X, train_y, epochs=100, batch_size=100, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    

        ### saving model to hard drive

    model_json = model.to_json()

    with open( var[0] + "_model.json", "w") as json_file:

        json_file.write(model_json)

    # serialize weights to HDF5

    model.save_weights(var[0] + "_model.h5")

    print("Saved model to disk")

    

    # plot history

#     pyplot.plot(history.history['loss'], label='train')

#     pyplot.plot(history.history['val_loss'], label='test')

#     pyplot.legend()

#     pyplot.show()



    test_X1 = test_X

    test_y1 = test_y



    test_X = test_X1

    test_y = test_y1



    test_X = test_X1

    test_y = test_y1



    yhat = model.predict(train_X)

   

    train_X = train_X.reshape((train_X.shape[0], n_hours*no_of_days))

    inv_yhat = concatenate((train_X[:, -no_of_outcome:], yhat), axis=1)



    inv_yhat = scaler.inverse_transform(inv_yhat)

    inv_yhat = inv_yhat[:,-1:]



    # invert scaling for actual

    train_y = train_y.reshape(len(train_y), train_y.shape[1])



    inv_y = concatenate((train_X[:, -no_of_outcome:], train_y), axis=1)

    inv_y = scaler.inverse_transform(inv_y)

    inv_y = inv_y[:,-1:]

    

    tmp_l = []

    tmp_l.append(var[0])

    for i in range(inv_y.shape[1]):

        rmse = sqrt(mean_squared_error(inv_y[:,i], inv_yhat[:,i]))

        tmp_l.append(rmse)

     

    # make a prediction

    yhat = model.predict(test_X)



#     print(test_X.shape[0])

    

    test_X = test_X.reshape((test_X.shape[0], n_hours*no_of_days))

    inv_yhat = concatenate((test_X[:, -no_of_outcome:], yhat), axis=1)



    inv_yhat = scaler.inverse_transform(inv_yhat)

    inv_yhat = inv_yhat[:,-1:]



    # invert scaling for actual

    test_y = test_y.reshape(len(test_y), test_y.shape[1])



    inv_y = concatenate((test_X[:, -no_of_outcome:], test_y), axis=1)

    inv_y = scaler.inverse_transform(inv_y)

    inv_y = inv_y[:,-1:]



    for i in range(inv_y.shape[1]):

        rmse = sqrt(mean_squared_error(inv_y[:,i], inv_yhat[:,i]))

        tmp_l.append(rmse)



    RMSE.append(tmp_l)



    acc = pd.DataFrame(inv_y[:,i], columns=['act'])



    acc['pred'] = inv_yhat[:,i]



    acc['diff'] = acc['act']-acc['pred']

    

    plt.figure(figsize=(20,5))

    sns.distplot(acc['diff'], kde=False, norm_hist=False)

    plt.show()

    

    %matplotlib inline

    fig, ax = plt.subplots(figsize=(20,10))

    sns.lineplot(y=acc['act'], x = range(len(test_X)), ax=ax, color='b', label = 'actual')

    sns.lineplot(y=acc['pred'], x = range(len(test_X)), ax=ax, color='r', label = 'predicted')

    ax.set_xticks(range(len(test_X)))

    ax.set_xticklabels(dt_lst[-len(test_X):], rotation=90)

#     sns.lineplot(y=acc['act'], ax=ax, color='b')

#     sns.lineplot(y=acc['pred'], ax=ax, color='r')

#     labels = ax.get_xticklabels()

    ax.lines[1].set_linestyle("--")

    ax.set_title(var[0]+"_act vs pred")

    ax.legend()

    plt.show()

    

    acc = acc.astype(dtype=int)

    pred_thres = np.full((20,2),0, dtype=float)

    for i in range(1, 20):

        ln = len(acc)

        pred = len(acc[(acc['diff']>-i) & (acc['diff'] < i)])

        pred_thres[i,0] = i

        pred_thres[i,1] = pred/ln



    fig, ax = plt.subplots(figsize=(20,10))

    sns.barplot(y=pred_thres[:,1], x = pred_thres[:,0], ax=ax)

    ax.set_title(var[0]+"_threshold distribution")

    plt.show()

    

print(pd.DataFrame(RMSE, columns=['feature', 'train', 'test']))