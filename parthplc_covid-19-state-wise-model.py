# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
state_wise = pd.read_csv("/kaggle/input/totalcases/state_wise_daily.csv")
state_wise.tail()
tamil_nadu = state_wise[state_wise.columns[:3]]
tamil_nadu.head()
df = tamil_nadu.pivot_table('TT', ['Date'], 'Status')
df.columns
df["Dates"] = df.index
df["Dates"] = pd.to_datetime(df["Dates"])

df = df.sort_values(by="Dates")

df_tn = df
df_tn.head()
df = state_wise[state_wise.columns[:2]]

df["MH"] = state_wise["MH"]

maha_rastra = df.pivot_table('MH', ['Date'], 'Status')

maha_rastra["Dates"] = maha_rastra.index

maha_rastra["Dates"] = pd.to_datetime(maha_rastra["Dates"])

maha_rastra = maha_rastra.sort_values(by="Dates")

df_mh = maha_rastra
df_mh.head()
df = state_wise[state_wise.columns[:2]]

df["TN"] = state_wise["TN"]

tamil_nadu = df.pivot_table('TN', ['Date'], 'Status')

tamil_nadu["Dates"] = tamil_nadu.index

tamil_nadu["Dates"] = pd.to_datetime(tamil_nadu["Dates"])

tamil_nadu = tamil_nadu.sort_values(by="Dates")

df_tn = tamil_nadu
df_tn.head()
df = state_wise[state_wise.columns[:2]]

df["GJ"] = state_wise["GJ"]

guja_rat = df.pivot_table('GJ', ['Date'], 'Status')

guja_rat["Dates"] = guja_rat.index

guja_rat["Dates"] = pd.to_datetime(guja_rat["Dates"])

guja_rat = guja_rat.sort_values(by="Dates")

df_gj = guja_rat
df_gj.head()
df = state_wise[state_wise.columns[:2]]

df["DL"] = state_wise["DL"]

delhi = df.pivot_table('DL', ['Date'], 'Status')

delhi["Dates"] = delhi.index

delhi["Dates"] = pd.to_datetime(delhi["Dates"])

delhi = delhi.sort_values(by="Dates")

df_dl = delhi
df_dl.head()
df = state_wise[state_wise.columns[:2]]

df["RJ"] = state_wise["RJ"]

rajas_than = df.pivot_table('RJ', ['Date'], 'Status')

rajas_than["Dates"] = rajas_than.index

rajas_than["Dates"] = pd.to_datetime(rajas_than["Dates"])

rajas_than = rajas_than.sort_values(by="Dates")

df_rj = rajas_than
df_rj.head()
df = state_wise[state_wise.columns[:2]]

df["UP"] = state_wise["UP"]

uttar_pradesh = df.pivot_table('UP', ['Date'], 'Status')

uttar_pradesh["Dates"] = uttar_pradesh.index

uttar_pradesh["Dates"] = pd.to_datetime(uttar_pradesh["Dates"])

uttar_pradesh = uttar_pradesh.sort_values(by="Dates")

df_up = uttar_pradesh
df_up.head()
df = state_wise[state_wise.columns[:2]]

df["MP"] = state_wise["MP"]

madhya_pradesh = df.pivot_table('MP', ['Date'], 'Status')

madhya_pradesh["Dates"] = madhya_pradesh.index

madhya_pradesh["Dates"] = pd.to_datetime(madhya_pradesh["Dates"])

madhya_pradesh = madhya_pradesh.sort_values(by="Dates")

df_mp = madhya_pradesh
df_mp.head()
df = state_wise[state_wise.columns[:2]]

df["WB"] = state_wise["WB"]

west_bengal = df.pivot_table('WB', ['Date'], 'Status')

west_bengal["Dates"] = west_bengal.index

west_bengal["Dates"] = pd.to_datetime(west_bengal["Dates"])

west_bengal = west_bengal.sort_values(by="Dates")

df_wb = west_bengal
df_wb.head()
df_mh.reset_index(drop=True, inplace=True)

df_tn.reset_index(drop=True, inplace=True)

df_gj.reset_index(drop=True, inplace=True)

df_rj.reset_index(drop=True, inplace=True)

df_up.reset_index(drop=True, inplace=True)

df_dl.reset_index(drop=True, inplace=True)

df_mp.reset_index(drop=True, inplace=True)

df_wb.reset_index(drop=True, inplace=True)
df_mh.reset_index(drop = True,inplace = True)

df_mh.set_index("Dates",inplace = True)

df_tn.reset_index(drop = True,inplace = True)

df_tn.set_index("Dates",inplace = True)

df_gj.reset_index(drop = True,inplace = True)

df_gj.set_index("Dates",inplace = True)

df_rj.reset_index(drop = True,inplace = True)

df_rj.set_index("Dates",inplace = True)

df_up.reset_index(drop = True,inplace = True)

df_up.set_index("Dates",inplace = True)

df_dl.reset_index(drop = True,inplace = True)

df_dl.set_index("Dates",inplace = True)

df_mp.reset_index(drop = True,inplace = True)

df_mp.set_index("Dates",inplace = True)

df_wb.reset_index(drop = True,inplace = True)

df_wb.set_index("Dates",inplace = True)
df_dl.head()
df_mh.head()
df_mh["Active"] = df_mh["Confirmed"] - df_mh["Deceased"] - df_mh["Recovered"]

df_tn["Active"] = df_tn["Confirmed"] - df_tn["Deceased"] - df_tn["Recovered"]

df_gj["Active"] = df_gj["Confirmed"] - df_gj["Deceased"] - df_gj["Recovered"]

df_dl["Active"] = df_dl["Confirmed"] - df_dl["Deceased"] - df_dl["Recovered"]

df_rj["Active"] = df_rj["Confirmed"] - df_rj["Deceased"] - df_rj["Recovered"]

df_up["Active"] = df_up["Confirmed"] - df_up["Deceased"] - df_up["Recovered"]

df_mp["Active"] = df_mp["Confirmed"] - df_mp["Deceased"] - df_mp["Recovered"]

df_wb["Active"] = df_wb["Confirmed"] - df_wb["Deceased"] - df_wb["Recovered"]
df_mh.head()
def active(df):

    df["Total_Active"] = 0

    for i in range(len(df)):

        df["Total_Active"][0] = df["Active"][0]

        if i>0:

            df["Total_Active"][i] = df["Total_Active"][i-1]+df["Active"][i] 
active(df_mh)

active(df_gj)

active(df_tn)

active(df_dl)

active(df_rj)

active(df_up)

active(df_mp)

active(df_wb)
df_mh.head()
from statsmodels.tsa.ar_model import AutoReg

from random import random

import math

# contrived dataset

data = list(df_mh["Total_Active"][:])

# fit model

model = AutoReg(data, lags=1)

model_fit = model.fit()

# make prediction

yhat_auto_mh = model_fit.predict(len(data), len(data)+60)

yhat_auto_mh =  [math.floor(i) for i in yhat_auto_mh]

print(yhat_auto_mh)
data[-5:]
project = pd.DataFrame()
project["Auto_Maharatra"] = yhat_auto_mh 
from statsmodels.tsa.ar_model import AutoReg

from random import random

# contrived dataset

data = list(df_gj["Total_Active"][:])

# fit model

model = AutoReg(data, lags=1)

model_fit = model.fit()

# make prediction

yhat_auto_gj = model_fit.predict(len(data), len(data)+60)

yhat_auto_gj =  [math.floor(i) for i in yhat_auto_gj]



print(yhat_auto_gj)

project["Auto_Gujarat"] = yhat_auto_gj 
from statsmodels.tsa.ar_model import AutoReg

from random import random

# contrived dataset

data = list(df_tn["Total_Active"][:])

# fit model

model = AutoReg(data, lags=1)

model_fit = model.fit()

# make prediction

yhat_auto_tn = model_fit.predict(len(data), len(data)+60)

yhat_auto_tn =  [math.floor(i) for i in yhat_auto_tn]



print(yhat_auto_tn)

project["Auto_TamilNadu"] = yhat_auto_tn 
from statsmodels.tsa.ar_model import AutoReg

from random import random

# contrived dataset

data = list(df_dl["Total_Active"][:])

# fit model

model = AutoReg(data, lags=1)

model_fit = model.fit()

# make prediction

yhat_auto_dl = model_fit.predict(len(data), len(data)+60)

yhat_auto_dl =  [math.floor(i) for i in yhat_auto_dl]



print(yhat_auto_dl)

project["Auto_Delhi"] = yhat_auto_dl 
from statsmodels.tsa.ar_model import AutoReg

from random import random

# contrived dataset

data = list(df_rj["Total_Active"][:])

# fit model

model = AutoReg(data, lags=1)

model_fit = model.fit()

# make prediction

yhat_auto_rj = model_fit.predict(len(data), len(data)+60)

yhat_auto_rj =  [math.floor(i) for i in yhat_auto_rj]



print(yhat_auto_rj)

project["Auto_Rajasthan"] = yhat_auto_rj 
from statsmodels.tsa.ar_model import AutoReg

from random import random

# contrived dataset

data = list(df_up["Total_Active"][:])

# fit model

model = AutoReg(data, lags=1)

model_fit = model.fit()

# make prediction

yhat_auto_up = model_fit.predict(len(data), len(data)+60)

yhat_auto_up =  [math.floor(i) for i in yhat_auto_up]



print(yhat_auto_up)

project["Auto_UttarPradesh"] = yhat_auto_up 
from statsmodels.tsa.ar_model import AutoReg

from random import random

# contrived dataset

data = list(df_mp["Total_Active"][:])

# fit model

model = AutoReg(data, lags=1)

model_fit = model.fit()

# make prediction

yhat_auto_mp = model_fit.predict(len(data), len(data)+60)

yhat_auto_mp =  [math.floor(i) for i in yhat_auto_mp]



print(yhat_auto_mp)

project["Auto_MadhyaPradesh"] = yhat_auto_mp 
from statsmodels.tsa.ar_model import AutoReg

from random import random

# contrived dataset

data = list(df_wb["Total_Active"][:])

# fit model

model = AutoReg(data, lags=1)

model_fit = model.fit()

# make prediction

yhat_auto_wb = model_fit.predict(len(data), len(data)+60)

yhat_auto_wb =  [math.floor(i) for i in yhat_auto_wb]



print(yhat_auto_wb)

project["Auto_WestBengal"] = yhat_auto_wb
project.head()
list(df_mh["Total_Active"][-5:])
#Example

# univariate lstm example

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.callbacks import EarlyStopping



# split a univariate sequence into samples

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix+10 > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+10]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)

 

# define input sequence

raw_seq = list(df_mh["Total_Active"][:])

# choose a number of time steps

n_steps = 5

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))

model.add(Dense(10))

model.compile(optimizer='adam', loss='mse')

# fit model

callback = EarlyStopping(monitor='loss', patience=3)



model.fit(X, y, epochs=10000, verbose=0, batch_size=5, callbacks=[callback])

# demonstrate prediction

x_input = array(list(df_mh["Total_Active"][-5:]))

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)
#predicting next forty days

def lstm_pred(list_a):

    yhat_lstm = []

    x_input = array(list_a)

    for i in range(6):

        x_input = x_input.reshape((1, n_steps, n_features))

        yhat = model.predict(x_input,verbose = 2)

        yhat_lstm = yhat_lstm + list(yhat[0])

        x_input = array(yhat_lstm[-5:])

    return (yhat_lstm)

yhat_lstm_mh = lstm_pred(list(df_mh["Total_Active"][-5:]))
yhat_lstm_mh = [math.floor(i) for i in yhat_lstm_mh]

project_lstm = pd.DataFrame()
project_lstm["Lstm_Maharatra"] = yhat_lstm_mh
#Example

# univariate lstm example

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.callbacks import EarlyStopping



# split a univariate sequence into samples

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix+10 > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+10]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)

 

# define input sequence

raw_seq = list(df_tn["Total_Active"][:])

# choose a number of time steps

n_steps = 5

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))

model.add(Dense(10))

model.compile(optimizer='adam', loss='mse')

# fit model

callback = EarlyStopping(monitor='loss', patience=3)



model.fit(X, y, epochs=10000, verbose=0, batch_size=5, callbacks=[callback])

# demonstrate prediction

x_input = array(list(df_tn["Total_Active"][-5:]))

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)

yhat_lstm_tn = lstm_pred(list(df_tn["Total_Active"][-5:]))

yhat_lstm_tn = [math.floor(i) for i in yhat_lstm_tn]

project_lstm["Lstm_TamilNadu"] = yhat_lstm_tn
project_lstm
#Example

# univariate lstm example

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.callbacks import EarlyStopping



# split a univariate sequence into samples

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix+10 > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+10]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)

 

# define input sequence

raw_seq = list(df_gj["Total_Active"][:])

# choose a number of time steps

n_steps = 5

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))

model.add(Dense(10))

model.compile(optimizer='adam', loss='mse')

# fit model

callback = EarlyStopping(monitor='loss', patience=3)



model.fit(X, y, epochs=10000, verbose=0, batch_size=5, callbacks=[callback])

# demonstrate prediction

x_input = array(list(df_gj["Total_Active"][-5:]))

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)

yhat_lstm_gj = lstm_pred(list(df_gj["Total_Active"][-5:]))

yhat_lstm_gj = [math.floor(i) for i in yhat_lstm_gj]

project_lstm["Lstm_Gujarat"] = yhat_lstm_gj
project_lstm
#Example

# univariate lstm example

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.callbacks import EarlyStopping



# split a univariate sequence into samples

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix+10 > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+10]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)

 

# define input sequence

raw_seq = list(df_dl["Total_Active"][:])

# choose a number of time steps

n_steps = 5

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))

model.add(Dense(10))

model.compile(optimizer='adam', loss='mse')

# fit model

callback = EarlyStopping(monitor='loss', patience=3)



model.fit(X, y, epochs=10000, verbose=0, batch_size=5, callbacks=[callback])

# demonstrate prediction

x_input = array(list(df_dl["Total_Active"][-5:]))

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)

yhat_lstm_dl = lstm_pred(list(df_dl["Total_Active"][-5:]))

yhat_lstm_dl = [math.floor(i) for i in yhat_lstm_dl]

project_lstm["Lstm_Delhi"] = yhat_lstm_dl
#Example

# univariate lstm example

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.callbacks import EarlyStopping



# split a univariate sequence into samples

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix+10 > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+10]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)

 

# define input sequence

raw_seq = list(df_up["Total_Active"][:])

# choose a number of time steps

n_steps = 5

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))

model.add(Dense(10))

model.compile(optimizer='adam', loss='mse')

# fit model

callback = EarlyStopping(monitor='loss', patience=3)



model.fit(X, y, epochs=10000, verbose=0, batch_size=5, callbacks=[callback])

# demonstrate prediction

x_input = array(list(df_up["Total_Active"][-5:]))

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)

yhat_lstm_up = lstm_pred(list(df_up["Total_Active"][-5:]))

yhat_lstm_up = [math.floor(i) for i in yhat_lstm_up]

project_lstm["Lstm_UttarPradesh"] = yhat_lstm_up
#Example

# univariate lstm example

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.callbacks import EarlyStopping



# split a univariate sequence into samples

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix+10 > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+10]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)

 

# define input sequence

raw_seq = list(df_mp["Total_Active"][:])

# choose a number of time steps

n_steps = 5

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))

model.add(Dense(10))

model.compile(optimizer='adam', loss='mse')

# fit model

callback = EarlyStopping(monitor='loss', patience=3)



model.fit(X, y, epochs=10000, verbose=0, batch_size=5, callbacks=[callback])

# demonstrate prediction

x_input = array(list(df_mp["Total_Active"][-5:]))

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)

yhat_lstm_mp = lstm_pred(list(df_mp["Total_Active"][-5:]))

yhat_lstm_mp = [math.floor(i) for i in yhat_lstm_mp]

project_lstm["Lstm_Madhya Pradesh"] = yhat_lstm_mp
#Example

# univariate lstm example

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.callbacks import EarlyStopping



# split a univariate sequence into samples

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix+10 > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+10]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)

 

# define input sequence

raw_seq = list(df_rj["Total_Active"][:])

# choose a number of time steps

n_steps = 5

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))

model.add(Dense(10))

model.compile(optimizer='adam', loss='mse')

# fit model

callback = EarlyStopping(monitor='loss', patience=3)



model.fit(X, y, epochs=10000, verbose=0, batch_size=5, callbacks=[callback])

# demonstrate prediction

x_input = array(list(df_rj["Total_Active"][-5:]))

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)

yhat_lstm_rj = lstm_pred(list(df_rj["Total_Active"][-5:]))

yhat_lstm_rj = [math.floor(i) for i in yhat_lstm_rj]

project_lstm["Lstm_Rajasthan"] = yhat_lstm_rj
#Example

# univariate lstm example

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.callbacks import EarlyStopping



# split a univariate sequence into samples

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix+10 > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+10]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)

 

# define input sequence

raw_seq = list(df_wb["Total_Active"][:])

# choose a number of time steps

n_steps = 5

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))

model.add(Dense(10))

model.compile(optimizer='adam', loss='mse')

# fit model

callback = EarlyStopping(monitor='loss', patience=3)



model.fit(X, y, epochs=10000, verbose=0, batch_size=5, callbacks=[callback])

# demonstrate prediction

x_input = array(list(df_wb["Total_Active"][-5:]))

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)

yhat_lstm_wb = lstm_pred(list(df_wb["Total_Active"][-5:]))

yhat_lstm_wb = [math.floor(i) for i in yhat_lstm_wb]

project_lstm["Lstm_WestBengal"] = yhat_lstm_wb
project_lstm.head()
# univariate bidirectional lstm example

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Bidirectional



# split a univariate sequence

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix+10 > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+10]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)



# define input sequence

raw_seq = list(df_mh["Total_Active"][:])

# choose a number of time steps

n_steps = 5

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))

model.add(Dense(10))

model.compile(optimizer='adam', loss='mse')

# fit model

callback = EarlyStopping(monitor='loss', patience=3)

model.fit(X, y, epochs=10000, verbose=0, batch_size=10, callbacks=[callback])

# demonstrate prediction

x_input = array(list(df_mh["Total_Active"][-5:]))

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)


def bilstm_pred(list_a):

    yhat_bilstm = []

    x_input = array(list_a)

    for i in range(6):

        x_input = x_input.reshape((1, n_steps, n_features))

        yhat = model.predict(x_input,verbose = 2)

        yhat_bilstm = yhat_bilstm + list(yhat[0])

        x_input = array(yhat_bilstm[-5:])

    return yhat_bilstm    
yhat_bilstm_mh = bilstm_pred(list(df_mh["Total_Active"][-5:]))

yhat_bilstm_mh = [math.floor(i) for i in yhat_bilstm_mh]

project_bilstm = pd.DataFrame()

project_bilstm["BILSTM_MH"] = yhat_bilstm_mh
project_bilstm.head()
from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Bidirectional



# split a univariate sequence

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix+10 > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+10]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)



# define input sequence

raw_seq = list(df_tn["Total_Active"][:])

# choose a number of time steps

n_steps = 5

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))

model.add(Dense(10))

model.compile(optimizer='adam', loss='mse')

# fit model

callback = EarlyStopping(monitor='loss', patience=3)

model.fit(X, y, epochs=10000, verbose=0, batch_size=10, callbacks=[callback])

# demonstrate prediction

x_input = array(list(df_tn["Total_Active"][-5:]))

x_input = x_input.reshape((1, n_steps, n_features))

yhat_bilstm_tn = bilstm_pred(list(df_tn["Total_Active"][-5:]))

yhat_bilstm_tn = [math.floor(i) for i in yhat_bilstm_tn]

project_bilstm["BILSTM_TN"] = yhat_bilstm_tn
project_bilstm.head()
from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Bidirectional



# split a univariate sequence

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix+10 > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+10]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)



# define input sequence

raw_seq = list(df_gj["Total_Active"][:])

# choose a number of time steps

n_steps = 5

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))

model.add(Dense(10))

model.compile(optimizer='adam', loss='mse')

# fit model

callback = EarlyStopping(monitor='loss', patience=3)

model.fit(X, y, epochs=10000, verbose=0, batch_size=10, callbacks=[callback])

# demonstrate prediction

x_input = array(list(df_gj["Total_Active"][-5:]))

x_input = x_input.reshape((1, n_steps, n_features))

yhat_bilstm_gj = bilstm_pred(list(df_gj["Total_Active"][-5:]))

yhat_bilstm_gj = [math.floor(i) for i in yhat_bilstm_gj]

project_bilstm["BILSTM_GJ"] = yhat_bilstm_gj
project_bilstm.head()
from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Bidirectional



# split a univariate sequence

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix+10 > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+10]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)



# define input sequence

raw_seq = list(df_dl["Total_Active"][:])

# choose a number of time steps

n_steps = 5

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))

model.add(Dense(10))

model.compile(optimizer='adam', loss='mse')

# fit model

callback = EarlyStopping(monitor='loss', patience=3)

model.fit(X, y, epochs=10000, verbose=0, batch_size=10, callbacks=[callback])

# demonstrate prediction

x_input = array(list(df_dl["Total_Active"][-5:]))

x_input = x_input.reshape((1, n_steps, n_features))

yhat_bilstm_dl = bilstm_pred(list(df_dl["Total_Active"][-5:]))

yhat_bilstm_dl = [math.floor(i) for i in yhat_bilstm_dl]

project_bilstm["BILSTM_DL"] = yhat_bilstm_dl
from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Bidirectional



# split a univariate sequence

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix+10 > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+10]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)



# define input sequence

raw_seq = list(df_up["Total_Active"][:])

# choose a number of time steps

n_steps = 5

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))

model.add(Dense(10))

model.compile(optimizer='adam', loss='mse')

# fit model

callback = EarlyStopping(monitor='loss', patience=3)

model.fit(X, y, epochs=10000, verbose=0, batch_size=10, callbacks=[callback])

# demonstrate prediction

x_input = array(list(df_up["Total_Active"][-5:]))

x_input = x_input.reshape((1, n_steps, n_features))

yhat_bilstm_up = bilstm_pred(list(df_up["Total_Active"][-5:]))

yhat_bilstm_up = [math.floor(i) for i in yhat_bilstm_up]

project_bilstm["BILSTM_UP"] = yhat_bilstm_up
from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Bidirectional



# split a univariate sequence

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix+10 > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+10]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)



# define input sequence

raw_seq = list(df_mp["Total_Active"][:])

# choose a number of time steps

n_steps = 5

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))

model.add(Dense(10))

model.compile(optimizer='adam', loss='mse')

# fit model

callback = EarlyStopping(monitor='loss', patience=3)

model.fit(X, y, epochs=10000, verbose=0, batch_size=10, callbacks=[callback])

# demonstrate prediction

x_input = array(list(df_mp["Total_Active"][-5:]))

x_input = x_input.reshape((1, n_steps, n_features))

yhat_bilstm_mp = bilstm_pred(list(df_mp["Total_Active"][-5:]))

yhat_bilstm_mp = [math.floor(i) for i in yhat_bilstm_mp]

project_bilstm["BILSTM_MP"] = yhat_bilstm_mp
from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Bidirectional



# split a univariate sequence

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix+10 > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+10]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)



# define input sequence

raw_seq = list(df_rj["Total_Active"][:])

# choose a number of time steps

n_steps = 5

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))

model.add(Dense(10))

model.compile(optimizer='adam', loss='mse')

# fit model

callback = EarlyStopping(monitor='loss', patience=3)

model.fit(X, y, epochs=10000, verbose=0, batch_size=10, callbacks=[callback])

# demonstrate prediction

x_input = array(list(df_rj["Total_Active"][-5:]))

x_input = x_input.reshape((1, n_steps, n_features))

yhat_bilstm_rj = bilstm_pred(list(df_rj["Total_Active"][-5:]))

yhat_bilstm_rj = [math.floor(i) for i in yhat_bilstm_rj]

project_bilstm["BILSTM_RJ"] = yhat_bilstm_rj
from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Bidirectional



# split a univariate sequence

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix+10 > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+10]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)



# define input sequence

raw_seq = list(df_wb["Total_Active"][:])

# choose a number of time steps

n_steps = 5

# split into samples

X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model

model = Sequential()

model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))

model.add(Dense(10))

model.compile(optimizer='adam', loss='mse')

# fit model

callback = EarlyStopping(monitor='loss', patience=3)

model.fit(X, y, epochs=10000, verbose=0, batch_size=10, callbacks=[callback])

# demonstrate prediction

x_input = array(list(df_wb["Total_Active"][-5:]))

x_input = x_input.reshape((1, n_steps, n_features))

yhat_bilstm_wb = bilstm_pred(list(df_wb["Total_Active"][-5:]))

yhat_bilstm_wb = [math.floor(i) for i in yhat_bilstm_wb]

project_bilstm["BILSTM_WB"] = yhat_bilstm_wb
project_bilstm.head()
project.head()
project['Date'] = pd.date_range(start='6/26/2020', periods=len(project), freq='D')
project.head()
project.set_index("Date",inplace = True)
project_lstm['Date'] = pd.date_range(start='6/26/2020', periods=len(project_lstm), freq='D')

project_lstm.set_index("Date",inplace = True)
project_bilstm['Date'] = pd.date_range(start='6/26/2020', periods=len(project_bilstm), freq='D')

project_bilstm.set_index("Date",inplace = True)
project.head()
project.to_csv("autoregtotalactive_states.csv")
project_lstm.head()
project_lstm.to_csv("lstmtotalactive.csv")

project_bilstm.head()
# project.to_csv("autoreg.csv")

# project_lstm.to_csv("lstm.csv")



project_bilstm.to_csv("bilstmtotalactive.csv")