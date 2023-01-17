import numpy as np

import pandas as pd

import seaborn as sns

from numpy.fft import *

import matplotlib.pyplot as plt

import matplotlib.style as style 

style.use('ggplot')



import os

print(os.listdir("../input"))
# Load data

X_train = pd.read_csv('../input/X_train.csv')

X_test = pd.read_csv('../input/X_test.csv')

target = pd.read_csv('../input/y_train.csv')
series_dict = {}

for series in (X_train['series_id'].unique()):

    series_dict[series] = X_train[X_train['series_id'] == series] 
# From: Code Snippet For Visualizing Series Id by @shaz13

def plotSeries(series_id):

    style.use('ggplot')

    plt.figure(figsize=(28, 16))

    print(target[target['series_id'] == series_id]['surface'].values[0].title())

    for i, col in enumerate(series_dict[series_id].columns[3:]):

        if col.startswith("o"):

            color = 'red'

        elif col.startswith("a"):

            color = 'green'

        else:

            color = 'blue'

        if i >= 7:

            i+=1

        plt.subplot(3, 4, i + 1)

        plt.plot(series_dict[series_id][col], color=color, linewidth=3)

        plt.title(col)
plotSeries(1)
# from @theoviel at https://www.kaggle.com/theoviel/fast-fourier-transform-denoising

def filter_signal(signal, threshold=1e3):

    fourier = rfft(signal)

    frequencies = rfftfreq(signal.size, d=20e-3/signal.size)

    fourier[frequencies > threshold] = 0

    return irfft(fourier)
# denoise train and test angular_velocity and linear_acceleration data

X_train_denoised = X_train.copy()

X_test_denoised = X_test.copy()



# train

for col in X_train.columns:

    if col[0:3] == 'ang' or col[0:3] == 'lin':

        # Apply filter_signal function to the data in each series

        denoised_data = X_train.groupby(['series_id'])[col].apply(lambda x: filter_signal(x))

        

        # Assign the denoised data back to X_train

        list_denoised_data = []

        for arr in denoised_data:

            for val in arr:

                list_denoised_data.append(val)

                

        X_train_denoised[col] = list_denoised_data

        

# test

for col in X_test.columns:

    if col[0:3] == 'ang' or col[0:3] == 'lin':

        # Apply filter_signal function to the data in each series

        denoised_data = X_test.groupby(['series_id'])[col].apply(lambda x: filter_signal(x))

        

        # Assign the denoised data back to X_train

        list_denoised_data = []

        for arr in denoised_data:

            for val in arr:

                list_denoised_data.append(val)

                

        X_test_denoised[col] = list_denoised_data

        
series_dict = {}

for series in (X_train_denoised['series_id'].unique()):

    series_dict[series] = X_train_denoised[X_train_denoised['series_id'] == series] 
plotSeries(1)
plt.figure(figsize=(24, 8))

plt.title('linear_acceleration_X')

plt.plot(X_train.angular_velocity_Z[128:256], label="original");

plt.plot(X_train_denoised.angular_velocity_Z[128:256], label="denoised");

plt.legend()

plt.show()
X_test.to_csv('X_test_denoised.csv', index=False)

X_train.to_csv('X_train_denoised.csv', index=False)