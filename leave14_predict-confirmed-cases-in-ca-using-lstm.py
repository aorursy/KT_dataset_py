# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import tensorflow as tf

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot

import seaborn as sns; sns.set()

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

import missingno as msno







# Configure visualisations

%matplotlib inline

mpl.style.use( 'ggplot' )

plt.style.use('fivethirtyeight')

sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.
filename = "/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv"

df = pd.read_csv(filename)



df
df.columns

#Obtaining the data 

on_data = df[df['Province/State'].isin(['Ontario'])]

on_data = on_data.iloc[:, 4:]
on_data
ca_data = df[df["Country/Region"] == 'Canada']
df_confirmed = pd.DataFrame(ca_data[ca_data.columns[4:]].sum(), columns=["Total confirm"])

df_confirmed
data_df=df_confirmed['Total confirm'].values

data_df=data_df.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0, 1))

confirmed_norm = scaler.fit_transform(data_df)

data_df
confirmed_norm
# split a univariate sequence into samples

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)

        y.append(seq_y)

    return np.array(X), np.array(y)
X, y = split_sequence(confirmed_norm, 4)

X
y
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.1,random_state=42)
X_train.shape
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

X_train.shape
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
#define the architecture of the model

def create_model():

  model = Sequential()

  model.add(LSTM(32,input_shape=(4,1),return_sequences=True,activation='relu'))

  model.add(LSTM(32,return_sequences=True))

  model.add(LSTM(64))

  model.add(Dense(1))

  return model
#train the model

model=create_model()

lr_reduce =tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=15,mode='auto')

model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae'])

LSTM_model = model.fit(X_train,Y_train, epochs=25, batch_size=1, validation_data=(X_test, Y_test),callbacks=[lr_reduce])

plt.plot(LSTM_model.history['loss'])

plt.plot(LSTM_model.history['val_loss'])

plt.title('LSTM BS -1 Epochs -25')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['train','validation'],loc='best')
plt.plot(LSTM_model.history['mae'])

plt.plot(LSTM_model.history['val_mae'])

plt.title('LSTM BS -1 Epochs -25')

plt.xlabel('Epochs')

plt.ylabel('mae')

plt.legend(['train','validation'],loc='best')
#prediction

n_steps = 4

# new_confirms = np.array([])

new_confirms = confirmed_norm.copy()

new_confirms = new_confirms.tolist()

new_confirms_inverse = data_df.copy()

new_confirms_inverse = new_confirms_inverse.tolist()



for i in range(14):

    X_input = new_confirms[-n_steps:]

    X_input = np.array(X_input)

    X_input = X_input.reshape((1, n_steps, 1))

    pred = model.predict(X_input, verbose=0)

    pred_arr = pred[0].tolist()

    pred_inverse = scaler.inverse_transform(pred)

    pred_arr_inverse = pred_inverse[0].tolist()

    pred_arr_inverse = [int(j) for j in pred_arr_inverse]

    new_confirms.append(pred_arr)

    new_confirms_inverse.append(pred_arr_inverse)

    

# X_input = new_confirms[-n_steps:]

# X_input = np.array(X_input)

# X_input = X_input.reshape((1, n_steps, 1))

# pred = model.predict(X_input, verbose=0)

# pred_arr = pred[0].tolist()

# pred_inverse = scaler.inverse_transform(pred)

# pred_arr_inverse = pred_inverse[0].tolist()

# pred_arr_inverse = [int(j) for j in pred_arr_inverse]

# new_confirms.append(pred_arr)

# new_confirms_inverse.append(pred_arr_inverse)
new_confirms_inverse
x1 = [i for i in range(len(new_confirms_inverse))]

plt.plot(x1[:-14], new_confirms_inverse[:-14])

plt.plot(x1[-14:], new_confirms_inverse[-14:])

plt.xlabel("# of days since the original date (1/22/20).")

plt.ylabel("# confirmed cases.")

plt.title('Prediction of confirmed cases for the following 14 days.')