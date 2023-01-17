

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv1D, Flatten, GlobalAveragePooling1D, InputLayer, Lambda

from tensorflow.keras import Sequential



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_path = '/kaggle/input/european-restaurant-profitability/restaurants.csv'

data = pd.read_csv(data_path)

data.head()
data = data.drop(['Company ID', 'Country', 'Archive standard coding'],axis=1)



data_n = data.drop(['Country code'],axis=1)

data.dtypes

data_norm = (data_n - data_n.min()) / (data_n.max() - data_n.min())

data = pd.concat((data_norm, data['Country code']), 1)

data.head()
X = np.array(data.drop(['ROE'], axis=1))

y = np.array(data.ROE)



train_size = int(len(data) * 0.8)

X_train, y_train = X[:train_size], y[:train_size]

X_test, y_test = X[train_size:], y[train_size:]



print("X_train shape:", X_train.shape)

print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)

print("y_test shape:", y_test.shape)
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(100).batch(32)

test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
model = Sequential([

                    InputLayer(input_shape=(16)),

                    Lambda(lambda x: tf.expand_dims(x, axis=-1)),

                    Conv1D(32,3, activation='relu'),

                    GlobalAveragePooling1D(),

                    Flatten(),

                    Dense(50, kernel_regularizer=tf.keras.regularizers.l2(l2=0.01), activation='relu'),

                    Dense(1)

])

model.summary()
model.compile(optimizer='adam',

              loss='mean_squared_error',

              metrics=['mae'])



history = model.fit(train_data, epochs=50, validation_data=test_data)
plt.style.use('ggplot')

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ax1.set_title('Loss by epochs')

ax1.plot(history.history['loss'], label='loss')

ax1.plot(history.history['val_loss'], label='valid_loss')

ax1.legend()



ax2.set_title('Mean absolute error by epochs')

ax2.plot(history.history['mae'], label='mae')

ax2.plot(history.history['val_mae'], label='val_mae')