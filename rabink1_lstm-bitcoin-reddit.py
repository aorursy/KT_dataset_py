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
ddf = pd.read_json('/kaggle/input/crypto-predict/data_1.json', lines=True)
ddf.head(1)
#Filtering Columns

def filter_col(df, cols):

    return df.drop(df.columns[[cols]], axis=1)



ddf= filter_col(ddf,[0,1,2,3,4,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41])
ddf.head(1)
#Changing to human readable datetime





def to_datetime(df):

    return pd.to_datetime(df['created_utc'],unit='s')



ddf['datetime']=to_datetime(ddf)
#Delete removed and deleted comments



ddf=ddf[ddf.body!='[removed]']

ddf=ddf[ddf.body!='[deleted]']

# Performing vader sentimental analysis



from nltk.sentiment.vader import SentimentIntensityAnalyzer

from pandas import Series 



analyzer = SentimentIntensityAnalyzer()

ddf[['compound','neg','neu','pos']] = ddf['body'].apply(lambda body: pd.Series(analyzer.polarity_scores(body)))

#Grouping the data on datetime and mean of the scores

def mean_grouping(df):

    return df.set_index('datetime').groupby(pd.Grouper(freq='D')).mean().dropna()



bitcoin_comment=mean_grouping(ddf)
#counting total number of comments per day

def comment_count(df):

    return df.set_index('datetime').resample('D').size()



bitcoin_comment['tot_comments']=comment_count(ddf)

# Mean,MAX,MIN for compound score

def min_max_mean(coin, df):

    print("Minumum compound ", coin," :" ,df['compound'].min())

    print("Maximum compound ", coin," :" ,df['compound'].max())

    print("Mean of compound ", coin," :" ,df['compound'].mean())

    print("\n")

    

min_max_mean('BIT',bitcoin_comment)
#saving to pickle 

bitcoin_comment.to_pickle('Bitcoin2017-18.pkl')
#Reading price data

prices = pd.read_csv("/kaggle/input/all-crypto-currencies/crypto-markets.csv" , parse_dates= ['date'])
prices.head(1)
#Making date as index

def date_index(df, coin):

    return df[df['symbol'] == coin].set_index('date')



pricesBTC = date_index(prices, 'BTC')
#Filtering data from 01-01-2018 to 09-11-2018



def filter_date(df, after, before):

    return df.loc[after:before]



pricesBTC = filter_date(pricesBTC, '2018-01-01','2018-11-09')
#scaling and adding some extra variables

from sklearn import preprocessing

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import MinMaxScaler



def operations(df):

    scaler = MinMaxScaler(feature_range=(-1, 1))

    df['prices'] = df[['open', 'close','high','low']].mean(axis=1)

    df['delta_day'] = df['high'] - df['low']

    df['fluctuation'] = 0

    df['fluctuation'] = df.delta_day.diff().fillna(0)

    df['norm_fluctuation'] = 0

    df['norm_fluctuation'] = scaler.fit_transform(df[['fluctuation']])

    df['pct_change']= df['prices'].pct_change()

    df['log_pct_change'] = np.log(df['prices'].astype('float64')/df['prices'].astype('float64').shift(1))

   

operations(pricesBTC)
pricesBTC.head(5)
#Saving dataset 

pricesBTC.to_pickle('pricesBTC2018-19.pkl')
btc = pd.read_pickle('/kaggle/input/kernel7769dea63e/Bitcoin2017-18.pkl')

pricebtc = pd.read_pickle('/kaggle/input/kernel7769dea63e/pricesBTC2018-19.pkl')
#Merge datasets

def merge(leftdf, rightdf):

    return pd.merge(leftdf,rightdf, how='inner', left_index=True, right_index=True)

    



mergebtc = merge(pricebtc, btc)



mergebtc.to_pickle('mergeBTC2018-19.pkl')
def corr_matrix(df):

    df = df[['prices','volume','delta_day', 'compound', 'tot_comments']]

    return df.corr(method='pearson')

    

corr_matrix(mergebtc)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.pairplot(mergebtc ,vars=['prices','volume','delta_day', 'compound', 'tot_comments'])
#loading data sets



btc=pd.read_pickle("/kaggle/input/kernel7769dea63e/mergeBTC2018-19.pkl")

btc.head()
# for selecting variables e.g All variables: df = select_var(ltc ,'prices','compound', 'numcomments')

def select_var(crypto, *vars):

    df = crypto

    if len(vars) == 4:

        df = df[[vars[0],vars[1],vars[2],vars[3]]]

    elif len(vars) == 3:

        df = df[[vars[0],vars[1],vars[2]]]

    elif len(vars) == 2:

        df = df[[vars[0],vars[1]]]

    else:

        df = df[[vars[0]]]

    return df



df = select_var(btc, 'prices')

df.head()
from sklearn import preprocessing

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import MinMaxScaler



from pandas import DataFrame

from pandas import concat

from pandas import Series

from pandas import Panel



from numpy import concatenate
values = df.values

values = values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))

scaled = scaler.fit_transform(values)



values[1]






n_days = 1

n_features = 1



# convert series to supervised learning by shifting t-1, t-2, t-3 depending on lag of days

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]

    df = DataFrame(data)

    cols, names = list(), list()

    #input sequence (t-n, ... t-1)

    for i in range(n_in, 0, -1):

        cols.append(df.shift(i))

        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    #forecast sequence (t, t+1, ... t+n)

    for i in range(0, n_out):

        cols.append(df.shift(-i))

        if i == 0:

            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]

        else:

            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    

    agg = concat(cols, axis=1)

    agg.columns = names

    # drop rows with NaN values

    if dropnan:

        agg.dropna(inplace=True)

    return agg





# frame as supervised learning

reframed = series_to_supervised(scaled, n_days,1)



print(reframed.shape)

print(reframed.head(), '\n')



if df.shape[1] != n_features:

    print('ERROR: n_features must match input variables')



values = reframed.values

train = values[:249, :]

test = values[249:, :]



#split into input and outputs

n_obs = n_days * n_features

train_X, train_y = train[:, :n_obs], train[:, -n_features]

test_X, test_y = test[:, :n_obs], test[:, -n_features]

print(train_X.shape, len(train_X), train_y.shape)



train_X = train_X.reshape((train_X.shape[0], n_days, n_features))

test_X = test_X.reshape((test_X.shape[0], n_days, n_features))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)



if df.shape[1] != n_features:

    print('ERROR: n_features must match input variables')



import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Activation

from keras.layers import Dropout

from keras.layers import LSTM

from keras.layers import Embedding

from keras.wrappers.scikit_learn import KerasRegressor

from keras import regularizers



import tensorflow as tf
model = Sequential()

model.add(LSTM(20, input_shape=(train_X.shape[1], train_X.shape[2])))

model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')

model.summary()

#Fitting the model and plotting loss





from matplotlib import pyplot

import matplotlib.pyplot as plt



from math import sqrt
history = model.fit(train_X, train_y, epochs=130, batch_size=72, validation_data=(test_X, test_y), verbose=1, shuffle=False)

pyplot.plot(history.history['loss'], label='train')

pyplot.plot(history.history['val_loss'], label='test')

pyplot.legend()

pyplot.show()

# make a prediction with model

yhat = model.predict(test_X)

test_X = test_X.reshape((test_X.shape[0], n_days*n_features))
# invert scaling for forecast

inv_yhat = concatenate((yhat, test_X[:, -(n_features-1):]), axis=1)

inv_yhat = scaler.inverse_transform(inv_yhat)

inv_yhat = inv_yhat[:,0]
# invert scaling for actual

test_y = test_y.reshape((len(test_y), 1))

inv_y = concatenate((test_y, test_X[:, -(n_features-1):]), axis=1)

inv_y = scaler.inverse_transform(inv_y)

inv_y = inv_y[:,0]

# calculate RMSE



testrmse = sqrt(mean_squared_error(inv_y, inv_yhat))



print('Test RMSE (Root Mean Squared Error): %.3f' % testrmse)

print('')



testmae = mean_absolute_error(inv_y, inv_yhat)

print('Test MAE (Mean Absolute Error): %.3f' % testmae)







def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



MAPE = mean_absolute_percentage_error(inv_y, inv_yhat )

print("Mean Absolute Percentage Error = ", MAPE)
#predicted vs real values

pyplot.plot(inv_yhat, label='predict')

pyplot.plot(inv_y, label='actual', alpha=0.5)

pyplot.legend()

pyplot.show()
d = {'real':inv_y, 'pred':inv_yhat}

pred = DataFrame(data=d)

pred['percentage change'] = (((pred['pred'] - pred['real'])/(pred['real']))*100)

pred['pred_change'] = pred.pred.diff().fillna(0)

pred['real_change'] = pred.real.diff().fillna(0)

print(pred)

def accuracy(df):

    count = 0

    for index, row in pred.iterrows():

        if row['pred_change'] < 0 and row['real_change'] < 0 or row['pred_change'] > 0 and row['real_change'] > 0:

            count += 1

    return count/len(df)*100



print('Accuracy of correct direction prediction (up/down): ', accuracy(pred))
def variance(df):

    count = 0

    percentage = 0

    for index, row in pred.iterrows():

        if row['pred_change'] < 0 and row['real_change'] > 0 or row['pred_change'] > 0 and row['real_change'] < 0:

            count += 1

            percentage += abs(row['percentage change'])

    return percentage/count



print('Variance of incorrect prediction how far off (up/down): ', variance(pred))
def f1_score(df):

    TN = 0

    TP = 0

    FN = 0

    FP = 0

    for index, row in pred.iterrows():

            if row['pred_change'] < 0 and row['real_change'] < 0: 

                TN += 1

            elif row['pred_change'] > 0 and row['real_change'] > 0:

                TP += 1

            elif row['pred_change'] < 0 and row['real_change'] > 0:

                FN += 1

            elif row['pred_change'] > 0 and row['real_change'] < 0:

                FP += 1



    precision = TP/(TP+FP)

    recall = TP/(TP+FN)

    f1 = 2*((recall*precision)/(recall + precision))

    return f1*100



print('The F1 score:',f1_score(df))