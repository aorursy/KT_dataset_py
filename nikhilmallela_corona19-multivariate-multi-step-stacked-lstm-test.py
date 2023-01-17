import numpy as np 

import matplotlib.pyplot as plt 

import matplotlib.colors as mcolors

import pandas as pd 

import random

import math

import time

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

import datetime

plt.style.use('seaborn')

%matplotlib inline 
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv", header=0, infer_datetime_format=True, parse_dates=['Date'])

test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv", header=0, infer_datetime_format=True, parse_dates=['Date'])

submission_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")

train_df.set_index(keys=['Date'], drop=False,inplace=True)

test_df.set_index(keys=['Date'], drop=False,inplace=True)

# print(train_df.head())

# print(train_df.tail())

# print(test_df.tail())

submission_df.tail()
train_df.rename(columns = {"Country_Region":"Country"},inplace=True)

test_df.rename(columns = {"Country_Region":"Country"},inplace=True)

len(train_df.Country.unique())

type(train_df)
train_df.dtypes
import datetime as dt



def to_integer(dt_time):

    date_string = dt.datetime.strptime(str(dt_time), '%Y-%m-%d %H:%M:%S')

    d = dt.datetime.strftime(date_string, '%Y%m%d')

    return int(d)
train_df["Date1"] = train_df["Date"].apply(lambda x: to_integer(x))
train_df
# train_df['Country/Region'].astype(str)

# clean_train_df = train_df.drop("Province/State", axis=1,inplace= True)

# clean_test_df = test_df.drop("Province/State", axis=1,inplace= True)

print(train_df)

print(test_df.dtypes)

# print(clean_test_df.dtypes)

# print(submission_df.dtypes)
# #since the dataset contain null values also 

# #count total rows in each column which contain null values

# print(clean_train_df.isna().sum())

# print(clean_test_df.isna().sum())

# print(submission_df.isna().sum())

train = train_df[1:-328]

test = train_df[-328:]

print(train.shape,test.shape)
# # sort the dataframe

# # train_df.sort_values(by='Country', axis=1, inplace=True)

# # set the index to be this and don't drop

# # train_df.set_index(keys=['Country'], drop=False,inplace=True)

# # get a list of names

# # names=train_df['Country'].unique().tolist()

# # now we can perform a lookup on a 'view' of the dataframe

# # k=0

# # for i in names:

# #     j=i

# #     j=train_df.loc[train_df.Country == i]

# #     print(i)

# # #     j.drop("Country",axis = 1,inplace=True)

# #     print(j.shape)

def dataPrep(train_df):

#     col = ["Date", "Lat", "Long", "ConfirmedCases", "Fatalities"]

#     inputs = pd.DataFrame(columns = col)

#     cols = ["ConfirmedCases", "Fatalities"]

#     targets = pd.DataFrame(columns = cols)

#     for u in range(train_df.shape[0]):

#         inputs.append(train_df[["Date", "Lat", "Long", "ConfirmedCases", "Fatalities"]])

#         targets.append(train_df[["ConfirmedCases", "Fatalities"]])

    from numpy import array

    inputs = []

    targets = []

    for u in range(train_df.shape[0]-8):

        if(train_df.iloc[u]["Country"]==train_df.iloc[u+7]["Country"]):

            inputs.append(np.array(train_df.iloc[u:u+7][["Date1", "ConfirmedCases", "Fatalities"]]).tolist())

            targets.append(np.array(train_df.iloc[u+7][["ConfirmedCases", "Fatalities"]]).tolist())

    return inputs,targets
# from numpy import array

# x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])

# np.vstack((x_input,[23, 32, 22]))

# x_input = x_input.reshape((1, 3, 3))

# x_input.shape
train_inputs,train_targets = dataPrep(train)

test_inputs,test_targets = dataPrep(test)

# train_inputs.reshape(1,3,3)

# train_inputs.shape
print(train_inputs[0:10],test_targets[0])

print(type(train_inputs))
from keras.callbacks import EarlyStopping

from keras.layers import RepeatVector

from keras.layers import TimeDistributed

from keras import optimizers





hidden_layers = 32

output_nodes = 2

input_nodes = 7

batch_size = 32

epochs = 70

verbose = 0

lr = 0.001

n_features = 3

print(n_features)
# model = Sequential()

# model.add(LSTM(hidden_layers,activation='tanh', batch_input_shape=[None, 7, input_nodes], return_sequences=True))

# model.add(LSTM(hidden_layers))

# model.add(Dense(output_nodes, activation="relu"))

# model.add(Dense(output_nodes))

# model.compile(optimizer='adam', loss='mse')



# #Naive LSTM

# import tensorflow as tf

# model = tf.keras.Sequential()

# model.add(tf.keras.layers.LSTM(hidden_layers, batch_input_shape=[None, 7, input_nodes], return_sequences=True))

# model.add(tf.keras.layers.LSTM(hidden_layers))

# model.add(tf.keras.layers.Dense(output_nodes, activation="sigmoid"))



# optimizer = tf.keras.optimizers.Adam(lr=lr)

# model.compile(loss="mean_squared_error", optimizer=optimizer)



import tensorflow as tf

model = tf.keras.Sequential()

model.add(tf.keras.layers.LSTM(100, activation='relu', return_sequences=True, input_shape=(input_nodes, n_features)))

# model.add(tf.keras.layers.RepeatVector(2))

model.add(tf.keras.layers.LSTM(100, activation='sigmoid'))

model.add(tf.keras.layers.Dense(output_nodes))

model.compile(optimizer='adam', loss='mse')



# model.add(tf.keras.layers.LSTM(200, activation='relu', input_shape=(input_nodes, n_features)))

# model.add(tf.keras.layers.RepeatVector(output_nodes))

# model.add(tf.keras.layers.LSTM(200, activation='relu', return_sequences=True))

# model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features)))

# model.compile(optimizer='adam', loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=5)

model1 = model.fit(train_inputs, train_targets,

                    batch_size=batch_size,

                      epochs=epochs,

                   validation_data=(test_inputs, test_targets),

                      callbacks = [early_stopping]

                      )

test_df["Date1"] = test_df["Date"].apply(lambda x: to_integer(x))
test_df
# results = []

# input_nodes = 3

# for i in range(30):

#     X = np.array(train_df[["Date1", "ConfirmedCases", "Fatalities"]])[-7:]

#     X = X.reshape(7, input_nodes)

#     print(X)

#     if test_df["Date"].min().timestamp() in train_df["Date"].values.tolist():

#         result = np.array(train_df[train_df["Date"] == test_df["Date"].min()][["Date", "Lat", "Long", "ConfirmedCases", "Fatalities"]])[0, 3:]

#     else:

#         result = model.predict(np.array(X).reshape(1, 7, input_nodes)).reshape(-1)

#         X = np.concatenate((X[1:], np.append(X[-1, :3], result).reshape(1, input_nodes)), axis=0)

# #     print(result)
all_df = train_df

maxlen = 7

input_number = 3

cases_max = train_df["ConfirmedCases"].max()

fatal_max = train_df["Fatalities"].max()
print(test_df.shape)

print(submission_df.shape)
results = []

for idx in test_df.groupby(["Province_State", "Country"]).count().index:

    test_df_on_idx = test_df[(test_df["Province_State"] == idx[0]) &

                             (test_df["Country"] == idx[1])]

#     print(test_df_on_idx)

    train_df_on_idx = train_df[(all_df["Country"] == idx[1]) &

                               (all_df["Province_State"] == idx[0])]

    inputs = np.array(train_df_on_idx[["Date1", "ConfirmedCases", "Fatalities"]])[-maxlen:]

#     print(inputs)

    inputs = inputs.reshape(maxlen, input_number)

    for day in range(102):

        if int(1000000000*(datetime.timedelta(days=day) + test_df_on_idx["Date"].min()).timestamp()) in train_df_on_idx["Date"].values.tolist():

            result = np.array(train_df_on_idx[train_df_on_idx["Date"] == (datetime.timedelta(days=day) + test_df_on_idx["Date"].min())][["Date", "ConfirmedCases", "Fatalities"]])[0, 1:]

        else:

            result = model.predict(np.array(inputs).reshape(1, maxlen, input_number)).reshape(-1)

            inputs = np.concatenate((inputs[1:], np.append(inputs[-1, :1], result).reshape(1, input_number)), axis=0)

        results.append([(result[0]), result[1]])

#         print(result)
len(results)
cases = []

fatals = []

for i in range(len(results)):

    n = results[i][0] 

    f = results[i][1]

    try:

        cases.append(int(n))

    except:

        cases.append(0)

    

    try:

        fatals.append(int(f))

    except:

        fatals.append(0)
submission_df.head()
submission_df["ConfirmedCases"] = cases[:-108]

submission_df["Fatalities"] = fatals[:-108]

submission_df
submission_df.set_index("ForecastId")


submission_df.to_csv("submission.csv")