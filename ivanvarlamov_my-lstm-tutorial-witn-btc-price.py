# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# First step, import libraries.

import numpy as np 

import pandas as pd 

from matplotlib import pyplot as plt
# Import the dataset and encode the date

df = pd.read_csv('/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')

df.head(3)
df['date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date

df.head(3)
group = df.groupby('date')

Real_Price = group['Weighted_Price'].mean()
# split data

prediction_days = 30

df_train= Real_Price[:len(Real_Price)-prediction_days]

df_test= Real_Price[len(Real_Price)-prediction_days:]
df_train
# Data preprocess

training_set = df_train.values

training_set = np.reshape(training_set, (len(training_set), 1))

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

training_set = sc.fit_transform(training_set)

X_train = training_set[0:len(training_set)-1]

y_train = training_set[1:len(training_set)]

X_train = np.reshape(X_train, (len(X_train), 1, 1))
# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM



# Initialising the RNN

regressor = Sequential()



# Adding the input layer and the LSTM layer

regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))



# units: Positive integer, dimensionality of the output space.

# activation: Activation function to use.

# Default: hyperbolic tangent (tanh).

# If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).



# Adding the output layer

regressor.add(Dense(units = 1))



# Compiling the RNN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')



# Fitting the RNN to the Training set

regressor.fit(X_train, y_train, batch_size = 5, epochs = 30)
from keras.utils.vis_utils import plot_model

plot_model(regressor, to_file='model_plot.png', show_shapes=True, show_layer_names=True, expand_nested=False)
regressor.summary()
# Making the predictions

test_set = df_test.values

inputs = np.reshape(test_set, (len(test_set), 1))

inputs = sc.transform(inputs)

inputs = np.reshape(inputs, (len(inputs), 1, 1))

predicted_BTC_price = regressor.predict(inputs)

predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)
# Visualising the results

plt.figure(figsize=(15,5), dpi=80, facecolor='w', edgecolor='k')

ax = plt.gca()  

plt.plot(test_set, color = 'red', label = 'Real BTC Price')

plt.plot(predicted_BTC_price, color = 'blue', label = 'Predicted BTC Price')

plt.title('BTC Price Prediction', fontsize=14)

df_test = df_test.reset_index()

x=df_test.index

labels = df_test['date']

plt.xticks(x, labels, rotation = 'vertical')

for tick in ax.xaxis.get_major_ticks():

    tick.label1.set_fontsize(14)

for tick in ax.yaxis.get_major_ticks():

    tick.label1.set_fontsize(14)

plt.xlabel('Time', fontsize=14)

plt.ylabel('BTC Price(USD)', fontsize=14)

plt.legend(loc=2, prop={'size': 14})

plt.show()