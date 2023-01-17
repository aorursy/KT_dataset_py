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
train_dataset = pd.read_csv("/kaggle/input/til2020-test/fashion-mnist_train.csv")
train_x = train_dataset.drop("label", axis='columns')

train_y = train_dataset['label']
train_y = train_y.to_numpy()

from tensorflow.keras.utils import to_categorical

y = to_categorical(train_y)
# simple normalization

train_x /= 255
x = train_x.to_numpy()

x = np.reshape(x, (x.shape[0], 28, 28))
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras import initializers

from tensorflow import keras

from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D,Dropout, Dense



inputs_1 = keras.Input(shape=(28, 28), name='input')

inputs = Conv1D(28,kernel_size=3, activation='relu', padding='same', name='conv_1')(inputs_1)

inputs = Dropout(0.5)(inputs)

inputs = Conv1D(56,kernel_size=3, activation='relu', padding='same', name='conv_2')(inputs)

inputs = Dropout(0.5)(inputs)



inputs = GlobalAveragePooling1D()(inputs)

outputs = Dense(10,activation="softmax", name='predictions')(inputs)



model = keras.Model(inputs=inputs_1, outputs=outputs)



es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

mc = ModelCheckpoint('base_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)





model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
hist = model.fit(x,y, epochs =300,  validation_split=0.2,callbacks=[es, mc])
test_dataset = pd.read_csv("/kaggle/input/til2020-test/test_dataset.csv")



test_x = test_dataset.drop('id', axis='columns')

test_x /= 255

test_x = test_x.to_numpy()

test_x = np.reshape(test_x, (test_x.shape[0], 28, 28))



test_y = model.predict(test_x)

test_y = np.argmax(test_y, axis=1)

new_csv = pd.DataFrame(data={'id': test_dataset['id'], 'Category': test_y})

new_csv.to_csv("submission.csv", index=False)