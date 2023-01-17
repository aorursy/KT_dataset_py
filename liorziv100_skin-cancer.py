# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/hmnist_28_28_L.csv")
temp_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)

train_df, val_df = train_test_split(temp_df, test_size=0.25, shuffle=True)

len(train_df), len(test_df), len(val_df)
train_labels = list(train_df['label'])

val_labels = list(val_df['label'])

test_labels = list(test_df['label'])



n_labels = len(set(list(df['label'])))



y_train = keras.utils.to_categorical(train_labels, n_labels)

y_val = keras.utils.to_categorical(val_labels, n_labels)

y_test = keras.utils.to_categorical(test_labels, n_labels)
def prep_data(df):

    data = df.drop(columns=['label']).values

    data = np.array(list(map(lambda x: x.reshape(28,28), data)))

    data = np.array(list(map(lambda x: x/255 , data)))

    data = data.reshape((len(data),28,28,1))

    

    return data



x_train = prep_data(train_df)

x_val = prep_data(val_df)

x_test = prep_data(test_df)

x_train.shape
from random import randint

plt.imshow(x_train[randint(0, len(x_train))].reshape(28,28), cmap="Greys")
import keras

from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
in_shape = (28, 28, 1)
model = keras.models.Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=in_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(n_labels, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



model.summary()
%reload_ext tensorboard.notebook

%tensorboard --logdir logs



tensorboard_callback = tf.keras.callbacks.TensorBoard("logs")
hist = model.fit(x_train, y_train,

         batch_size=16,

         epochs=30,

         verbose=1,

         callbacks=[tensorboard_callback],

         validation_data=(x_val, y_val))