# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from matplotlib import rcParams

rcParams['figure.figsize']=10,6

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.recurrent import LSTM

from keras.models import Sequential

import time

from sklearn.metrics import mean_squared_error

from matplotlib import pyplot

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset=pd.read_csv("../input/rainfall in india 1901-2015.csv",encoding = "ISO-8859-1")

dataset.dtypes
groups = dataset.groupby('SUBDIVISION')['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','NOV','DEC']

data=groups.get_group(('BIHAR'))

data.head()
data=data.melt(['YEAR']).reset_index()

data.head()

df= data[['YEAR','variable','value']].reset_index().sort_values(by=['YEAR','index'])

df.head()
df.columns=['INDEX','YEAR','Month','avg_rainfall']
df.head()
d={'JAN':1,'FEB':2,'MAR' :3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,

   'OCT':10,'NOV':11,'DEC':12}

df['Month']=df['Month'].map(d)

df.head(12)
df['Date']=pd.to_datetime(df.assign(Day=1).loc[:,['YEAR','Month','Day']])

df.head(12)
cols=['avg_rainfall']

dataset=df[cols]

dataset.head()
series=dataset

series.head()
series.shape
pyplot.figure(figsize=(20,6))

pyplot.plot(series.values)

pyplot.show()
# Get the raw data values from the pandas data frame.

data_raw = series.values.astype("float32")



# We apply the MinMax scaler from sklearn

# to normalize data in the (0, 1) interval.

scaler = MinMaxScaler(feature_range = (0, 1))

dataset = scaler.fit_transform(data_raw)



# Print a few values.

dataset[0:5]
# Using 60% of data for training, 40% for validation.

TRAIN_SIZE = 0.80



train_size = int(len(dataset) * TRAIN_SIZE)

test_size = len(dataset) - train_size

train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

print("Number of entries (training set, test set): " + str((len(train), len(test))))
# FIXME: This helper function should be rewritten using numpy's shift function. See below.

def create_dataset(dataset, window_size = 1):

    data_X, data_Y = [], []

    for i in range(len(dataset) - window_size - 1):

        a = dataset[i:(i + window_size), 0]

        data_X.append(a)

        data_Y.append(dataset[i + window_size, 0])

    return(np.array(data_X), np.array(data_Y))
# Create test and training sets for one-step-ahead regression.

window_size = 1

train_X, train_Y = create_dataset(train, window_size)

test_X, test_Y = create_dataset(test, window_size)

print("Original training data shape:")

print(train_X.shape)



# Reshape the input data into appropriate form for Keras.

train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))

test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

print("New training data shape:")

print(train_X.shape)
def fit_model(train_X, train_Y, window_size = 1):

    model = Sequential()

    

    model.add(LSTM(2000,activation = 'tanh', inner_activation = 'hard_sigmoid', input_shape = (1, window_size)))

    model.add(Dropout(0.2))

    model.add(Dense(500))

    model.add(Dropout(0.4))

    model.add(Dense(500))

    model.add(Dropout(0.4))

    model.add(Dense(400))

    model.add(Dropout(0.4))

    model.add(Dense(1, activation = 'linear'))

    model.compile(loss = "mean_squared_error", 

                  optimizer = "adam")

    model.fit(train_X, 

              train_Y, 

              epochs = 10, 

              batch_size = 64, 

              )

    

    return(model)



# Fit the first model.

model1 = fit_model(train_X, train_Y, window_size)
import math

def predict_and_score(model, X, Y):

    # Make predictions on the original scale of the data.

    pred = scaler.inverse_transform(model.predict(X))

    # Prepare Y data to also be on the original scale for interpretability.

    orig_data = scaler.inverse_transform([Y])

    # Calculate RMSE.

    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))

    return(score, pred)



rmse_train, train_predict = predict_and_score(model1, train_X, train_Y)

rmse_test, test_predict = predict_and_score(model1, test_X, test_Y)



print("Training data score: %.2f RMSE" % rmse_train)

print("Test data score: %.2f RMSE" % rmse_test)
# Start with training predictions.

train_predict_plot = np.empty_like(dataset)

train_predict_plot[:, :] = np.nan

train_predict_plot[window_size:len(train_predict) + window_size, :] = train_predict



# Add test predictions.

test_predict_plot = np.empty_like(dataset)

test_predict_plot[:, :] = np.nan

test_predict_plot[len(train_predict) + (window_size * 2) + 1:len(dataset) - 1, :] = test_predict



# Create the plot.

plt.figure(figsize = (18, 8))

plt.plot(scaler.inverse_transform(dataset), label = "True value",color='red')

plt.plot(train_predict_plot, label = "Training set prediction",color='yellow')

plt.plot(test_predict_plot, label = "Test set prediction")

plt.xlabel("Months")





plt.legend()

plt.show()
test_predict
train_predict