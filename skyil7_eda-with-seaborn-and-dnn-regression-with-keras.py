import warnings

warnings.filterwarnings(action='ignore')

import pandas as pd

import pandas_profiling

import seaborn as sns

sns.set()

import numpy as np
data = pd.read_csv('../input/sejongai-challenge-pretest-2/2020.AI.bike-train.csv', index_col=0)

test_data = pd.read_csv('../input/sejongai-challenge-pretest-2/2020.AI.bike-test.csv', index_col=0)

submit = pd.read_csv('../input/sejongai-challenge-pretest-2/2020.AI.bike-submission.csv')

data.head()
pp = pandas_profiling.ProfileReport(data, progress_bar=False)

pp
import matplotlib.pyplot as plt

corr = data.corr()

plt.figure(figsize=(15,12))

sns.heatmap(data.corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
data = data.drop('hour_bef_precipitation', axis=1)

test_data = test_data.drop('hour_bef_precipitation', axis=1)
data.head()
x = data.drop('count', axis=1) # count column is what we are interested.

y = data['count']



from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=777) # split x, y data into 80% of (train x, y), and 20% of (validation x, y)

print(x_train.shape)

print(x_val.shape)
from sklearn.preprocessing import MinMaxScaler

scale=MinMaxScaler()

x_train=scale.fit_transform(x_train) # we takes min(x) and max(x) from train data only.

x_val=scale.transform(x_val) # normalize using min(train_x), max(train_x) not min(val_x), max(val_x)
import tensorflow as tf

from keras import models

from keras import layers

from keras import regularizers



with tf.device('/device:GPU:0'):

    model = models.Sequential() # Keras DNN Model

    model.add(layers.Dense(64, activation='relu', input_shape=(8, ), kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001)))

    model.add(layers.Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001)))

    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(1, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001)))

    model.compile(optimizer='nadam', loss='mse', metrics=['mae']) # We takes Mean Squared Error(MSE) for our loss function and use ADAM Optimizer.



model.summary()
epoch = 1500

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epoch, batch_size=16, verbose=0) # Train Model



loss = history.history['loss'][30:] # Extract Loss history

val_loss = history.history['val_loss'][30:] # We don't use first 30 loss values for clean plot. (First 30 losses are too big)



plt.plot(range(1, len(loss)+1), loss, label='train loss')

plt.plot(range(1, len(loss)+1), val_loss, label='validation loss')



plt.xlabel('Epochs')

plt.legend()

plt.show()
x_test = scale.transform(test_data)

pred = model.predict(x_test)

pd.DataFrame(pred).head()
sns.distplot(pred)
submit['id'] = np.array(submit["count"], dtype=np.int)

submit.head()
submit['count'] = pred

submit.head()
submit.to_csv('submit.csv', index=False, header=True) # save DataFrame as csv file