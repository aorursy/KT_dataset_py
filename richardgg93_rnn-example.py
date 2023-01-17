# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf # RNN developing

from sklearn import preprocessing

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score



import matplotlib.pyplot as plt # drawing, visualization

import seaborn as sns # drawing, visualization



import datetime

from datetime import datetime as dt



from kaggle.competitions import twosigmanews



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print(tf.__version__)
# Retreive the environment of the competition

# You can only call make_env() once, so don't lose it!

env = twosigmanews.make_env()

print('Data loaded!')
# Retrieve all training data

(market_train_df, news_train_df) = env.get_training_data()

print("Fetching training data finished... ")

print('Data obtained!')
#### Market data analysis

# Types of the columns

#print(market_train_df.dtypes)

market_train_df.head()
# News data analysis

# Types of the columns

#print(news_train_df.dtypes)

news_train_df.head()
# Minor changes on dates so they match the day

market_train_df['time'] = market_train_df['time'].dt.tz_convert(None).dt.normalize()

news_train_df['time'] = news_train_df['time'].dt.tz_convert(None).dt.normalize()
# Lets remove some market columns

cols_to_keep_market = ["time", "assetName", "returnsOpenNextMktres10"]

market_train_df = market_train_df[cols_to_keep_market]

market_train_df.head()
# Lets summarize the news

news_train_df["sent_rel"] = news_train_df["urgency"] * news_train_df["relevance"] * (news_train_df["sentimentPositive"]-news_train_df["sentimentNegative"]) 

cols_to_keep_news = ["time", "assetName", "sent_rel"]

news_train_df = news_train_df[cols_to_keep_news]

# Convert news of same day and asset to one row only

#d = {'sent_rel': 'mean', 'name': 'first', 'amount': 'sum'}

d_aggs = {'sent_rel' : 'mean'}

news_train_df=news_train_df[["time", "assetName"]+list(d_aggs.keys())].groupby(['time', 'assetName'], as_index=False).aggregate(d_aggs)

news_train_df.head()
# Merge market and news data

df_final = market_train_df.merge(news_train_df, how = 'left', on = ['time', 'assetName'])

# Replace NaNs

df_final.fillna({'sent_rel':df_final["sent_rel"].mean()}, inplace=True)

print(len(df_final))

df_final.head()
santander_names = [n for n in np.unique(df_final.assetName) if "Santander" in n]

print(santander_names)

santander_df = df_final[df_final.assetName == 'Banco Santander SA']

plt.plot(santander_df.time, santander_df["returnsOpenNextMktres10"])

plt.show()

plt.plot(santander_df.time, santander_df["sent_rel"])

plt.show()
facebook_names = [n for n in np.unique(df_final.assetName) if "Face" in n]

print(facebook_names)

facebook_df = df_final[df_final.assetName == 'Facebook Inc']

plt.plot(facebook_df.time, facebook_df["returnsOpenNextMktres10"])

plt.show()

plt.plot(facebook_df.time, facebook_df["sent_rel"])

plt.show()
apple_names = [n for n in np.unique(df_final.assetName) if "Apple" in n]

print(apple_names)

apple_df = df_final[df_final.assetName == 'Apple Inc']

plt.plot(apple_df.time, apple_df["returnsOpenNextMktres10"])

plt.show()

plt.plot(apple_df.time, apple_df["sent_rel"])

plt.show()
oracle_names = [n for n in np.unique(df_final.assetName) if "Oracle" in n]

print(oracle_names)

oracle_df = df_final[df_final.assetName == 'Oracle Corp']

plt.plot(oracle_df.time, oracle_df["returnsOpenNextMktres10"])

plt.show()

plt.plot(oracle_df.time, oracle_df["sent_rel"])

plt.show()
features = ["returnsOpenNextMktres10", "sent_rel"]

target = "returnsOpenNextMktres10"
# DEFINE THE MODEL

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.LSTM(16, unroll=True, return_sequences=True))

model.add(tf.keras.layers.LSTM(16, unroll=True))

model.add(tf.keras.layers.Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
# SELECT COMPANY DATA TO TRAIN WITH

aux = oracle_df.copy()

# SCALE THE VARIABLES TO HAVE MEAN 0 AND STD DEV 1

# OTHERWISE IT TAKES PLENTY OF TIME TO TRAIN AND IT STARTS WITH THE VALUES CLOSER TO 0

scaler_features = preprocessing.StandardScaler().fit(aux[features])

scaler_target = preprocessing.StandardScaler().fit(aux[target].values.reshape(-1,1))

aux[features] = scaler_features.transform(aux[features])

plt.plot(aux.time, aux[features[0]])

plt.show()

plt.plot(aux.time, aux[features[1]])

plt.show()

# SPLIT INTO TRAIN AND TEST (UP TO 2016 TRAIN, LATER TEST)

train = aux[aux.time < np.datetime64('2016-01-01')]

test = aux[aux.time >= np.datetime64('2016-01-01')]
# DEFINE THE PERIOD TO LOOK BACK FOR THE RNN

PERIOD = 50



# Preprocess train data

train_X=[]

train_y=[]

train_times=[]

for i in range(PERIOD, train.shape[0]):

    X, y, time = train.iloc[i-PERIOD:i,:][features].values, train.iloc[i,:][target], train.iloc[i,:].time

    train_X.append(X)

    train_y.append(y)

    train_times.append(time)

train_X = np.array(train_X)

train_y = np.array(train_y)

# reshape input to be [samples, time steps, features]

train_X = np.reshape(train_X, (train_X.shape[0], PERIOD, len(features)))



# Preprocess test data

test_X=[]

test_y=[]

test_times=[]

for i in range(PERIOD, test.shape[0]):

    X, y, time = test.iloc[i-PERIOD:i,:][features].values, test.iloc[i,:][target], test.iloc[i,:].time

    test_X.append(X)

    test_y.append(y)

    test_times.append(time)

test_X = np.array(test_X)

test_y = np.array(test_y)

# reshape input to be [samples, time steps, features]

test_X = np.reshape(test_X, (test_X.shape[0], PERIOD, len(features)))
# TRAIN THE MODEL

model.fit(train_X, train_y, batch_size=12, epochs=5, verbose=2, shuffle=False)     
# MAKE PREDICTIONS

trainPredictions = scaler_target.inverse_transform(model.predict(train_X))

testPredictions = scaler_target.inverse_transform(model.predict(test_X))

# accuracy

test_y2 = [0 if y < 0 else 1 for y in scaler_target.inverse_transform(test_y)]

testPredictions2 = [0 if y < 0 else 1 for y in testPredictions]

# scores

print("MAE:", mean_absolute_error(scaler_target.inverse_transform(test_y), testPredictions))

print("MSE:", mean_squared_error(scaler_target.inverse_transform(test_y), testPredictions))

print("Accuracy: ", accuracy_score(test_y2, testPredictions2))

# plots

plt.plot(train_times, scaler_target.inverse_transform(train_y))

plt.plot(train_times, trainPredictions)

plt.show()

plt.plot(train_times[-20:], scaler_target.inverse_transform(train_y[-20:]))

plt.plot(train_times[-20:], trainPredictions[-20:])

plt.show()

plt.plot(test_times, scaler_target.inverse_transform(test_y))

plt.plot(test_times, testPredictions)

plt.show()

plt.plot(test_times[-20:], scaler_target.inverse_transform(test_y[-20:]))

plt.plot(test_times[-20:], testPredictions[-20:])

plt.show()
# TEST MODEL ON FACEBOOK

aux = facebook_df.copy()

scaler_features = preprocessing.StandardScaler().fit(aux[features])

scaler_target = preprocessing.StandardScaler().fit(aux[target].values.reshape(-1,1))

aux[features] = scaler_features.transform(aux[features])

test=aux

# Preprocess test data

test_X=[]

test_y=[]

test_times=[]

for i in range(PERIOD, test.shape[0]):

    X, y, time = test.iloc[i-PERIOD:i,:][features].values, test.iloc[i,:][target], test.iloc[i,:].time

    test_X.append(X)

    test_y.append(y)

    test_times.append(time)

test_X = np.array(test_X)

test_y = np.array(test_y)

test_X = np.reshape(test_X, (test_X.shape[0], PERIOD, len(features)))

testPredictions = scaler_target.inverse_transform(model.predict(test_X))

test_y2 = [0 if y < 0 else 1 for y in scaler_target.inverse_transform(test_y)]

testPredictions2 = [0 if y < 0 else 1 for y in testPredictions]

# results

print("MAE:", mean_absolute_error(scaler_target.inverse_transform(test_y), testPredictions))

print("MSE:", mean_squared_error(scaler_target.inverse_transform(test_y), testPredictions))

print("Accuracy: ", accuracy_score(test_y2, testPredictions2))

plt.plot(test_times, scaler_target.inverse_transform(test_y))

plt.plot(test_times, testPredictions)

plt.show()

plt.plot(test_times[-20:], scaler_target.inverse_transform(test_y[-20:]))

plt.plot(test_times[-20:], testPredictions[-20:])

plt.show()
# TEST MODEL ON APPLE

aux = apple_df.copy()

scaler_features = preprocessing.StandardScaler().fit(aux[features])

scaler_target = preprocessing.StandardScaler().fit(aux[target].values.reshape(-1,1))

aux[features] = scaler_features.transform(aux[features])

test=aux

# Preprocess test data

test_X=[]

test_y=[]

test_times=[]

for i in range(PERIOD, test.shape[0]):

    X, y, time = test.iloc[i-PERIOD:i,:][features].values, test.iloc[i,:][target], test.iloc[i,:].time

    test_X.append(X)

    test_y.append(y)

    test_times.append(time)

test_X = np.array(test_X)

test_y = np.array(test_y)

test_X = np.reshape(test_X, (test_X.shape[0], PERIOD, len(features)))

testPredictions = scaler_target.inverse_transform(model.predict(test_X))

test_y2 = [0 if y < 0 else 1 for y in scaler_target.inverse_transform(test_y)]

testPredictions2 = [0 if y < 0 else 1 for y in testPredictions]

# results

print("MAE:", mean_absolute_error(scaler_target.inverse_transform(test_y), testPredictions))

print("MSE:", mean_squared_error(scaler_target.inverse_transform(test_y), testPredictions))

print("Accuracy: ", accuracy_score(test_y2, testPredictions2))

plt.plot(test_times, scaler_target.inverse_transform(test_y))

plt.plot(test_times, testPredictions)

plt.show()

plt.plot(test_times[-20:], scaler_target.inverse_transform(test_y[-20:]))

plt.plot(test_times[-20:], testPredictions[-20:])

plt.show()
# modelCNN = tf.keras.models.Sequential()

# modelCNN.add(tf.keras.layers.Conv1D(64, 3, activation='relu'))

# modelCNN.add(tf.keras.layers.Conv1D(64, 3, activation='relu'))

# modelCNN.add(tf.keras.layers.Conv1D(64, 3, activation='relu'))

# modelCNN.add(tf.keras.layers.GlobalAveragePooling1D())

# modelCNN.add(tf.keras.layers.Dense(10, activation='sigmoid'))

# modelCNN.add(tf.keras.layers.Dense(1))

# modelCNN.compile(loss='mean_squared_error', optimizer='adam')
# ### SCALE THE VARIABLE TO HAVE MEAN 0 AND STD DEV 1

# ### OTHERWISE IT TAKES PLENTY OF TIME TO TRAIN AND IT STARTS WITH THE VALUES CLOSER TO 0

# aux = market_train_df_santander.copy()

# scaler = preprocessing.StandardScaler().fit(aux[target].values.reshape(-1,1))

# aux[target] = scaler.transform(aux[target].values.reshape(-1,1))

# plt.plot(aux.time, aux[target])

# # CONVERT DATES TO HAVE ONLY THE DAY AND NO UTC

# # SPLIT INTO TRAIN AND TEST (UP TO 2016 TRAIN, LATER tEST)

# aux['time'] = aux['time'].dt.tz_convert(None).dt.normalize()

# train = aux[aux.time < np.datetime64('2016-01-01')]

# test = aux[aux.time >= np.datetime64('2016-01-01')]
# # DEFINE THE PERIOD TO LOOK BACK FOR THE RNN

# PERIOD = 50



# # Preprocess train data

# train_X=[]

# train_y=[]

# train_times=[]

# for i in range(PERIOD, train.shape[0]):

#     X, y, time = train.iloc[i-PERIOD:i,:][target], train.iloc[i,:][target], train.iloc[i,:].time

#     train_X.append(X)

#     train_y.append(y)

#     train_times.append(time)

# train_X = np.array(train_X)

# train_y = np.array(train_y)

# # reshape input to be [samples, time steps, features]

# train_X = np.reshape(train_X, (train_X.shape[0], PERIOD, 1))



# # Preprocess test data

# test_X=[]

# test_y=[]

# test_times=[]

# for i in range(PERIOD, test.shape[0]):

#     X, y, time = test.iloc[i-PERIOD:i,:][target], test.iloc[i,:][target], test.iloc[i,:].time

#     test_X.append(X)

#     test_y.append(y)

#     test_times.append(time)

# test_X = np.array(test_X)

# test_y = np.array(test_y)

# # reshape input to be [samples, time steps, features]

# test_X = np.reshape(test_X, (test_X.shape[0], PERIOD, 1))
# TRAIN THE MODEL

# modelCNN.fit(train_X, train_y, batch_size=12, epochs=1, verbose=2, shuffle=False) 
# # MAKE PREDICTIONS

# trainPredictions = scaler.inverse_transform(modelCNN.predict(train_X))

# testPredictions = scaler.inverse_transform(modelCNN.predict(test_X))

# plt.plot(train_times, scaler.inverse_transform(train_y))

# plt.plot(train_times, trainPredictions)

# plt.show()

# plt.plot(train_times[20:30], scaler.inverse_transform(train_y[20:30]))

# plt.plot(train_times[20:30], trainPredictions[20:30])

# plt.show()

# plt.plot(test_times, scaler.inverse_transform(test_y))

# plt.plot(test_times, testPredictions)

# plt.show()

# plt.plot(test_times[20:30], scaler.inverse_transform(test_y[20:30]))

# plt.plot(test_times[20:30], testPredictions[20:30])

# plt.show()