# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import keras

import pandas as pd

import numpy as np

from matplotlib import pyplot

from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

from datetime import datetime



standard_scaler = StandardScaler()

minmax_scaler = MinMaxScaler()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.0001)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test_df = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

train_df = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test_df = test_df.drop(['id'], axis=1)

#test_df = minmax_scaler.fit_transform(test_df)

test_df
Y_train_df = train_df['label']

Y_train_df = keras.utils.np_utils.to_categorical(Y_train_df)

Y_train_df.shape
X_train_df = train_df.drop(columns="label")

#X_train_df = minmax_scaler.fit_transform(X_train_df)

X_train_df.shape
np.random.seed(7)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_df, Y_train_df, test_size=0.10, random_state=7)
def create_model(train_data):

    # create model

    model = keras.models.Sequential()

    model.add(keras.layers.Dense(50000, input_dim=train_data.shape[1], kernel_initializer=keras.initializers.glorot_normal(seed=None)))

    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dropout(0.35))

    model.add(keras.layers.Dense(2000, kernel_initializer=keras.initializers.glorot_normal(seed=70)))

    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dropout(0.35))

    model.add(keras.layers.Dense(10))

    model.add(keras.layers.Activation("softmax"))

    # Compile model

    model.compile(optimizer ='rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy']) 

    return model
model = create_model(X_train_df)
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=170,batch_size=500, callbacks=[es, mc, learning_rate_reduction])

#history = model.fit(X_train_df, Y_train_df, epochs=40,batch_size=500, callbacks=[es, mc])
# plot training history

pyplot.plot(history.history['loss'], label='train')

pyplot.plot(history.history['val_loss'], label='test')

pyplot.legend()

pyplot.show()

best_model = keras.models.load_model('best_model.h5')
answer = model.predict(test_df)

#answer = best_model.predict(test_df)

answer = answer.astype(int)

y = np.argmax(answer, axis=-1)

y
frames_answer = pd.DataFrame({"id": pd.DataFrame(y).index.values,

                             "label": y

                            })

frames_answer
frames_answer.to_csv('submission.csv', index=False)