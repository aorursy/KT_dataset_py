

import tensorflow as tf

import numpy as np

import tensorflow.keras as keras

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_char = pd.read_csv('../input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data.csv').astype('float32')

data_char.rename(columns={'0':'label'}, inplace=True)
X_char = data_char.drop('label',axis = 1)

y_char = data_char['label']
X_train, X_test, y_train, y_test = train_test_split(X_char,y_char)
standard_scaler = MinMaxScaler()

standard_scaler.fit(X_train)
X_train = standard_scaler.transform(X_train)

X_test = standard_scaler.transform(X_test)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')



y_train = tf.keras.utils.to_categorical(y_train)

y_test = tf.keras.utils.to_categorical(y_test)
model=keras.models.Sequential([keras.layers.Conv2D(32,3,activation='relu', input_shape=[28,28,1]),

                               keras.layers.Conv2D(64, (3, 3), activation='relu'),

                               keras.layers.MaxPooling2D(pool_size=2),

                               keras.layers.Dropout(.4),

                               keras.layers.Flatten(),

                               keras.layers.Dense(128, activation='relu'),

                               keras.layers.Dense(26, activation='softmax'),

])
model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, batch_size=120)
history_df=pd.DataFrame(history.history)



history_df.plot(figsize = (8,8))

plt.grid(True)

plt.gca().set_ylim(0,1.2)

plt.show()
score = model.evaluate(X_test,y_test)



print("The score is : ", score[1])
model.save('model.h5')