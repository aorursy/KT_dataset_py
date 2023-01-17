import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, LSTM, Dropout

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.metrics import MeanSquaredLogarithmicError

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/daily-sun-spot-data-1818-to-2019/sunspot_data.csv')

df.head()
missing_vals = df.isnull().sum()

missing_vals
df.describe()
corr_df = df.corr()

corr_df
plt.figure(figsize=(17,10))

sns.heatmap(corr_df, cmap='coolwarm')
df_train = df[df['Year']<2000]

df_test = df[df['Year']>=2000]



spots_train = df_train['Number of Sunspots'].tolist()

spots_test = df_test['Number of Sunspots'].tolist()



print("Training set has {} observations.".format(len(spots_train)))

print("Test set has {} observations.".format(len(spots_test)))
def create_sequence(seq, obs):

    x = []

    y = []

    for i in range(len(obs)-size_sample):

        #print(i)

        window = obs[i:(i+size_sample)]

        after_window = obs[i+size_sample]

        window = [[x] for x in window]

        #print("{} - {}".format(window,after_window))

        x.append(window)

        y.append(after_window)

        

    return np.array(x),np.array(y)

    

    

size_sample = 15

x_train,y_train = create_sequence(size_sample,spots_train)

x_test,y_test = create_sequence(size_sample,spots_test)



print("Shape of training set: {}".format(x_train.shape))

print("Shape of test set: {}".format(x_test.shape))
x_train[0:10]
model = Sequential()

model.add(LSTM(128, dropout=0.0, recurrent_dropout=0.0, input_shape = (None,1)))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(32, activation = 'relu'))

model.add(Dense(1, activation = 'softmax'))





model.compile(optimizer = 'adam', loss = tf.keras.losses.MeanSquaredLogarithmicError())

early_stopping = EarlyStopping(monitor='val_loss', min_delta = 1e-3,

                               patience = 5, verbose = 1, 

                               mode='auto', restore_best_weights=True)



model.fit(x_train, y_train, validation_data = [x_test, y_test],

         callbacks = [early_stopping],

         epochs = 500, 

         verbose = 2)
from sklearn.metrics import mean_squared_log_error, mean_squared_error



y_pred = model.predict(x_test)

score = np.sqrt(mean_squared_log_error(y_pred, y_test))

mean_score = np.sqrt(mean_squared_error(y_pred, y_test))

print('The RMSLE value of {} is obtained '.format(score))

print('The RMSE value of {} is obtained '.format(mean_score))