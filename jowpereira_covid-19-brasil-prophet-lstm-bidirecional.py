# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv", index_col = 'Id')

train[train['Country_Region'] == 'Brazil'].tail(2)
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

copy_test = test[test['Country_Region'] == 'Brazil'].copy()
brasil_train = train[train['Country_Region'] == 'Brazil']

brasil_train[['Country_Region', 'Date', 'ConfirmedCases', 'Fatalities']].head()
fig,ax = plt.subplots(figsize=(15,7))



ax.plot(brasil_train['Date'], brasil_train[['ConfirmedCases']], color="b", marker="o")



ax.set_xlabel("tempo",fontsize=14)



ax.set_ylabel("infectados",color="b",fontsize=14)



ax2=ax.twinx()



ax2.plot(brasil_train['Date'], brasil_train[['Fatalities']],color="r",marker="o")

ax2.set_ylabel("mortos",color="r",fontsize=14)

plt.show()
width = 0.50

ConfirmedCases = brasil_train['ConfirmedCases'].to_numpy()

Fatalities     = brasil_train['Fatalities'].to_numpy()

Date           = brasil_train['Date'].to_numpy()



fig = plt.figure(figsize=(20,10))

ax = fig.add_axes([0,0,1,1])

ax.bar(Date, ConfirmedCases, width, color='b')

ax.bar(Date, Fatalities, width,bottom=ConfirmedCases, color='r')

ax.legend(labels=['ConfirmedCases', 'Fatalities'])

 

plt.show()
from fbprophet import Prophet



train_f = pd.DataFrame()

train_f['ds'] = brasil_train['Date']

train_f['y']  = brasil_train['Fatalities']
train_c = pd.DataFrame()

train_c['ds'] = brasil_train['Date']

train_c['y']  = brasil_train['ConfirmedCases']
model_prophet_f =  Prophet(changepoint_prior_scale=0.5)

model_prophet_f.fit(train_f)

future = model_prophet_f.make_future_dataframe(periods=28, freq= 'D')

prophet_out_f = model_prophet_f.predict(future)
model_prophet_c =  Prophet(changepoint_prior_scale=0.5)

model_prophet_c.fit(train_c)

future = model_prophet_c.make_future_dataframe(periods=28, freq= 'D')

prophet_out_c = model_prophet_c.predict(future)
fig1 = model_prophet_f.plot(prophet_out_f)
fig1 = model_prophet_c.plot(prophet_out_c)
fig2 = model_prophet_f.plot_components(prophet_out_f)
fig2 = model_prophet_c.plot_components(prophet_out_c)
from fbprophet.plot import add_changepoints_to_plot

fig = model_prophet_f.plot(prophet_out_f)

a = add_changepoints_to_plot(fig.gca(), model_prophet_f, prophet_out_f)
from fbprophet.plot import add_changepoints_to_plot

fig = model_prophet_c.plot(prophet_out_c)

a = add_changepoints_to_plot(fig.gca(), model_prophet_c, prophet_out_c)
fig,ax = plt.subplots(figsize=(15,7))



ax.plot(prophet_out_c[['yhat']], color="b", marker="o")



ax.set_xlabel("tempo",fontsize=5)



ax.set_ylabel("infectados",color="b",fontsize=14)



ax2=ax.twinx()



ax2.plot(prophet_out_f[['yhat']],color="r",marker="o")

ax2.set_ylabel("mortos",color="r",fontsize=14)

plt.show()
from __future__ import absolute_import

import numpy as np

import keras as k

import pandas as pd



from abc import ABC, abstractmethod



class LSTMModelBase(ABC):



    def __init__(self):

        self.__model = self.__create_model()



    def __create_model(self):

        return k.models.Sequential()

    

    def getModel(self):

        return self.__model

    

    @abstractmethod

    def fit(self):

        pass



    def predict(self, x):

        return self.__model.predict(x)



    def evaluate(self):

        pass



    def save(self, file_path):

        self.__model.save(file_path)



    def load(self, file_path):

        self.__model = k.models.load_model(file_path)



    @abstractmethod

    def create_network(self):

        pass



    # split a univariate sequence into samples

    def split_sequence(self, sequence, n_steps_in, n_steps_out):

        X, y = list(), list()

        for i in range(len(sequence)):

            # find the end of this pattern

            end_ix = i + n_steps_in

            out_end_ix = end_ix + n_steps_out

            # check if we are beyond the sequence

            if out_end_ix > len(sequence):

                break

            # gather input and output parts of the pattern

            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]

            X.append(seq_x)

            y.append(seq_y)

        return np.array(X), np.array(y)



    def rolling_window(self, df, window):

        shape = df.shape[:-1] + (df.shape[-1] - window + 1, window)

        strides = df.strides + (df.strides[-1],)

        return np.lib.stride_tricks.as_strided(df, shape=shape, strides=strides)

    

    def train_test_split(self, df, fator):

        return df[:int(df.shape[0]*fator)], df[int(df.shape[0]*(fator)):]
class Bidirecional(LSTMModelBase):

    def __init__(self, n_steps_in=2, n_steps_out=1, n_features=1):

        super(Bidirecional, self).__init__()

        self.__n_steps_in   = n_steps_in

        self.__n_steps_out  = n_steps_out

        self.__n_features   = n_features

        self.__n_windows    = self.__n_steps_in*self.__n_features



    def create_network(self):

        super().getModel().add(k.layers.wrappers.Bidirectional(k.layers.recurrent.LSTM(200, activation='relu', return_sequences=False), input_shape=(self.__n_steps_in, self.__n_features)))

        super().getModel().add(k.layers.core.Dropout(0.5))

        super().getModel().add(k.layers.core.Dense(self.__n_steps_out))

        super().getModel().compile(optimizer='adam', loss='mse')

        print('<< building Bidirecional >>')

        

    def fit(self, X, y, epochs, verbose):

        return super().getModel().fit(X, y, epochs=epochs, verbose=verbose, shuffle=False)

    

    def split_sequence(self, train):

        X, y = super().split_sequence(train, self.__n_steps_in, self.__n_steps_out)

        X = X.reshape((X.shape[0], X.shape[1], self.__n_features))

        return X, y

    

    def reshape(self, pred):

        return pred.reshape((1, self.__n_steps_in, self.__n_features))

    

    def getWindows(self):

        return self.__n_windows
series = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv", index_col = 'Id')

df = series[series['Country_Region'] == 'Brazil']

df[['Country_Region', 'Date', 'ConfirmedCases', 'Fatalities']].tail(2)
max(df['Date'])


df = pd.DataFrame(df.fillna('NA').groupby(['Country_Region','Date'])['Fatalities', 'ConfirmedCases'].sum().reset_index()).sort_values(['Country_Region','Date'])



df_new_deaths = pd.DataFrame(df.fillna('NA').groupby(['Country_Region','Date'])['Fatalities'].sum().reset_index()).sort_values(['Country_Region','Date'])

df_new_confirmed_cases = pd.DataFrame(df.fillna('NA').groupby(['Country_Region','Date'])['ConfirmedCases'].sum().reset_index()).sort_values(['Country_Region','Date'])



df_new_deaths.Fatalities = df_new_deaths.Fatalities.diff().fillna(0)

df_new_deaths = df_new_deaths.loc[df_new_deaths['Date'] == (df_new_deaths['Date']),['Country_Region','Fatalities']]



df_new_confirmed_cases.ConfirmedCases = df_new_confirmed_cases.ConfirmedCases.diff().fillna(0)

df_new_confirmed_cases = df_new_confirmed_cases.loc[df_new_confirmed_cases['Date'] == (df_new_confirmed_cases['Date']),['Country_Region','ConfirmedCases']]



df.Fatalities = df.Fatalities + df_new_deaths.Fatalities

df.ConfirmedCases = df.ConfirmedCases + df_new_confirmed_cases.ConfirmedCases
bidirecional_f   = Bidirecional(2,10,1)

bidirecional_c   = Bidirecional(2,10,1)

df         = df.set_index(df['Date'])

df         = df.sort_index()

motos      = df['Fatalities']

infectados = df['ConfirmedCases']
train_f, _ = bidirecional_f.train_test_split(motos, 1)

# split into samples

X_f, y_f = bidirecional_f.split_sequence(train_f)

#

bidirecional_f.create_network()

history_f = bidirecional_f.fit(X_f, y_f, 150, 0)
train_c, _ = bidirecional_c.train_test_split(infectados, 1)

# split into samples

X_c, y_c = bidirecional_c.split_sequence(train_c)

#

bidirecional_c.create_network()

history_c = bidirecional_c.fit(X_c, y_c, 150, 0)
plt.figure(figsize=(10,5))

plt.title('Loss')

plt.plot(history_f.history['loss'], label='train f')

plt.plot(history_c.history['loss'], label='train c')

plt.legend()

plt.show()
pred = motos.iloc[-2:].values

pred = bidirecional_f.reshape(pred)

predict_out_f = bidirecional_f.predict(pred)
pred = infectados.iloc[-2:].values

pred = bidirecional_c.reshape(pred)

predict_out_c = bidirecional_c.predict(pred)
fig,ax = plt.subplots(figsize=(15,7))



ax.plot(predict_out_c.flatten(), color="b", marker="o")



ax.set_xlabel("tempo",fontsize=14)



ax.set_ylabel("infectados",color="b",fontsize=14)



ax2=ax.twinx()



ax2.plot(predict_out_f.flatten(),color="r",marker="o")

ax2.set_ylabel("mortos",color="r",fontsize=14)

plt.show()
'''submit_df = pd.DataFrame()

submit_df['ForecastId']                     = copy_test['ForecastId']

submit_df['ConfirmedCases_prophet']         = pd.DataFrame(prophet_out_c)

submit_df['ConfirmedCases_rnnBidirecional'] = pd.DataFrame(predict_out_c.flatten())

submit_df['Fatalities_prophet']             = pd.DataFrame(prophet_out_f)

submit_df['Fatalities_rnnBidirecional']     = pd.DataFrame(predict_out_f.flatten())

submit_df.info()'''
'''submission_df.to_csv("submission.csv")'''