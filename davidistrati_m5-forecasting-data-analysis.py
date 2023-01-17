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
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from pandas_profiling import ProfileReport



from scipy import signal

from scipy.signal import butter, deconvolve

import pywt
df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

df.head()
dt = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

dt.head()
dates = [d for d in df.columns if 'd_' in d]

print('dates in the training set: ', dates[:5], ' ... ', dates[-5:])
ids = sorted(list(set(df['id'])))



all_sales = df[dates].sum(axis=0).values



x1_sales = (df['store_id'] == 'CA_2').values

x1_sales = np.where(x1_sales == True)

x1_sales = df.iloc[x1_sales[0]][dates].sum(axis=0).values



x2_sales = df.loc[df['id'] == ids[2]].set_index('id')[dates].values[0]



fig, ax = plt.subplots(3, figsize=(15,15))

ax[0].plot([x for x in range(len(all_sales))], all_sales, 'tab:green')

ax[0].set_title('All sales / day')

ax[0].set(xlabel='days', ylabel='sales')



ax[1].plot([x for x in range(len(x1_sales))], x1_sales, 'tab:red')

ax[1].set_title('Sales of a store / day')

ax[1].set(xlabel='days', ylabel='sales')



ax[2].plot([x for x in range(len(x2_sales))], x2_sales, 'tab:blue')

ax[2].set_title('Sales of an item / day')

ax[2].set(xlabel='days', ylabel='sales')



fig.tight_layout()

fig.show()
from scipy.fftpack import fft, ifft
def maxN(elements, n):

    return sorted(elements, reverse=True)[:n]

def apply_fft(data, n):

    fourier = fft(data)

    power = [int((abs(x) ** (2 / fourier.shape[0])) * 1000) for x in fourier]

    limit  = min(maxN(power, n))

    for i in range(fourier.shape[0]):

        if (power[i] < limit):

            fourier[i] = 0

    inv_fourier = ifft(fourier).real

    return inv_fourier



fft_sales_small = apply_fft(x2_sales, 5)

fft_sales_medium = apply_fft(x2_sales, 10)

fig, ax = plt.subplots(2, figsize=(15,10))

ax[0].plot([x for x in range(len(fft_sales_small))], fft_sales_small, 'coral')

ax[0].set_title('FFT with high noise reduction for the second graph above')

ax[0].set(xlabel='days', ylabel='sales')



ax[1].plot([x for x in range(len(fft_sales_medium))], fft_sales_medium, 'crimson')

ax[1].set_title('FFT with medium noise reduction for the second graph above')

ax[1].set(xlabel='days', ylabel='sales')



fig.tight_layout()

fig.show()
rolling_mean1 = pd.DataFrame({'y':x1_sales})

rolling_mean1 = rolling_mean1.y.rolling(window=14).mean()



rolling_mean2 = pd.DataFrame({'y':x2_sales})

rolling_mean2 = rolling_mean2.y.rolling(window=20).mean()



rolling_mean3 = pd.DataFrame({'y':all_sales})

rolling_mean3 = rolling_mean3.y.rolling(window=20).mean()



print('The graphs below are the first 3 graphs with MA noise reduction')



fig, ax = plt.subplots(3, figsize=(15,15))

ax[0].plot([x for x in range(len(rolling_mean3))], rolling_mean3, 'coral')

ax[0].set_title('Moving averages noise reduction')

ax[0].set(xlabel='days', ylabel='sales')



ax[1].plot([x for x in range(len(rolling_mean1))], rolling_mean1, 'crimson')

ax[1].set_title('Moving averages noise reduction')

ax[1].set(xlabel='days', ylabel='sales')



ax[2].plot([x for x in range(len(rolling_mean2))], rolling_mean2, 'tomato')

ax[2].set_title('Moving averages noise reduction')

ax[2].set(xlabel='days', ylabel='sales')



fig.tight_layout()

fig.show()
def maddest(d, axis=None):

    return np.mean(np.absolute(d - np.mean(d, axis)), axis)



def denoise_signal(x, wavelet='db4', level=1):

    coeff = pywt.wavedec(x, wavelet, mode="per")

    sigma = (1/0.45) * maddest(coeff[-level])



    uthresh = sigma * np.sqrt(2*np.log(len(x)))

    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])



    return pywt.waverec(coeff, wavelet, mode='per')



y_w1 = denoise_signal(all_sales)

y_w2 = denoise_signal(x1_sales)

y_w3 = denoise_signal(x2_sales)



print('The graphs below are the first 3 graphs with Wave denoising')



fig, ax = plt.subplots(3, figsize=(15,15))

ax[0].plot([x for x in range(len(y_w1))], y_w1, 'coral')

ax[0].set_title('Wave denoising noise reduction')

ax[0].set(xlabel='days', ylabel='sales')



ax[1].plot([x for x in range(len(y_w2))], y_w2, 'crimson')

ax[1].set_title('Wave denoising noise reduction')

ax[1].set(xlabel='days', ylabel='sales')



ax[2].plot([x for x in range(len(y_w3))], y_w3, 'tomato')

ax[2].set_title('Wave denoising noise reduction')

ax[2].set(xlabel='days', ylabel='sales')



fig.tight_layout()

fig.show()
item_1 = df.iloc[1].values[6:]

item_1_pred = dt.iloc[1].values[1:]

fig, ax = plt.subplots(1, figsize=(15,5))

ax.plot(([x for x in range(len(item_1))]+[x for x in range(len(item_1_pred))]), np.hstack((item_1,item_1_pred)), 'blue')

ax.axvspan(len(item_1), (len(item_1)+len(item_1_pred)), color='red', alpha=0.4)

ax.set(xlabel='days', ylabel='sales')

ax.set_title('We need to predict the sales in the red line')

!pip install pmdarima

from pmdarima import auto_arima
model = auto_arima(item_1[-300:-14], seasonal=True, m=14)
fig, ax = plt.subplots(1, figsize=(15,5))

ax.plot([x for x in range(300)], item_1[-300:], 'blue', label='Item sales')

ax.plot([x for x in range(300+len(item_1_pred))], np.hstack((model.predict_in_sample(),model.predict(14), model.predict(14+len(item_1_pred))[14:])), 'orange', label='Sarima prediction')

ax.axvspan((300-14), 300, color='green', alpha=0.4, label='test set')

ax.axvspan(300, 300+len(item_1_pred), color='red', alpha=0.2, label='prediction')

ax.legend(loc="upper left")

ax.set(xlabel='days', ylabel='sales')

ax.set_title('Iteam sales and sarima prediction')

fig.show()
from sklearn.metrics import mean_squared_error

def measure_mse(actual, predicted):

    return mean_squared_error(actual, predicted)
print('the score (mse)  of the model is: ', measure_mse(item_1[-14:],model.predict(14)))
from math import sqrt

from keras.models import Sequential

from keras.layers import Dense, LSTM

from keras.layers import Flatten

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D
def train_test_split(data, n_test):

    return data[:-n_test], data[-n_test:]



def series_to_supervised(data, n_in, n_out=1):

    df = pd.DataFrame(data)

    cols = list()



    for i in range(n_in, 0, -1):

        cols.append(df.shift(i))



    for i in range(0, n_out):

        cols.append(df.shift(-i))



    agg = pd.concat(cols, axis=1)



    agg.dropna(inplace=True)

    return agg.values



def prepare_dataset(train, n_input):

    data = series_to_supervised(train, n_input)

    train_x, train_y = data[:, :-1], data[:, -1]

    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))

    return train_x, train_y

    

def model_fit(train, config):

    n_input, n_filters, n_kernel, n_epochs, n_batch = config

    train_x, train_y = prepare_dataset(train, n_input)

    

def model_predict(model, history, config):

    

    n_input, _, _, _, _ = config

    

    x_input = np.array(history[-n_input:]).reshape((1, n_input, 1))

    

    yhat = model.predict(x_input, verbose=0)

    return yhat[0]



def predict_next_n(model, history, config, n):

    n_input, _, _, _, _ = config

    pred = []

    for i in range(n):

        x_input = np.array(history[-n_input:]).reshape((1, n_input, 1))



        yhat = model.predict(x_input, verbose=0)

        pred.append(yhat[0][0])

        history.append(yhat[0][0])

    return pred



def walk_forward_validation(data, n_test, cfg):

    predictions = list()



    train, test = train_test_split(data, n_test)



    model = model_fit(train, cfg)



    history = [x for x in train]



    for i in range(len(test)):

        yhat = model_predict(model, history, cfg)

        predictions.append(yhat)

        history.append(test[i])

    error = measure_mse(test, predictions)

    print(' > %.3f' % error)

    return [error, model]



def repeat_evaluate(data, config, n_test, n_repeats=1):

    for _ in range(n_repeats):

        return_list = walk_forward_validation(data, n_test, config)

        scores = return_list[0]

    return scores, return_list[1]





def prepare_dataset_no_y(train, n_input):

    data_prep = series_to_supervised(train, n_input)

    data_prep = data_prep.reshape((data_prep.shape[0], data_prep.shape[1], 1))

    return data_prep



def model_fit(train, config):

    n_input, n_filters, n_kernel, n_epochs, n_batch = config

    train_x, train_y = prepare_dataset(train, n_input)

    

    model = Sequential()

    model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu', input_shape=(n_input, 1), padding='same'))

    model.add(LSTM(128, activation='relu'))

    model.add(Dense(64, activation='relu'))

#     model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'))

#     model.add(MaxPooling1D(pool_size=2))

#     model.add(Flatten())

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)

    return model
data = item_1[-300:]

n_input = 7

n_test = 14

config = [n_input, 256, 3, 400, 300]

scores, model_cnn = repeat_evaluate(data, config, n_test)

print('the score (mse)  of the model is: ',scores)
test_x = series_to_supervised(data, n_in=(n_input-1), n_out=1)

test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))



x = np.arange(n_input, dtype=int)



pred = predict_next_n(model_cnn, data.tolist(), config, len(item_1_pred))



fig, ax = plt.subplots(1, figsize=(15,5))

ax.plot(data, color = 'blue')

ax.plot(np.vstack((np.zeros_like(x).reshape(-1,1),model_cnn.predict(test_x).reshape(-1,1), np.array(pred).reshape(-1,1))), color='orange')

ax.axvspan((len(data)-14), len(data), color='green', alpha=0.4, label='test set')

ax.axvspan((len(data)), len(data)+len(item_1_pred), color='red', alpha=0.2, label='prediction')

ax.legend()