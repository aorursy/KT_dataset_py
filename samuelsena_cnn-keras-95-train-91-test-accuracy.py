# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
import tensorflow as tf

from keras.utils import to_categorical

import numpy as np

import pandas as pd



train_data = pd.read_csv('../input/fashion-mnist_train.csv')

test_data = pd.read_csv('../input/fashion-mnist_test.csv')



img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)



# Train data

train_x = np.array(train_data.iloc[:, 1:])

train_y = to_categorical(np.array(train_data.iloc[:, 0]))



# Test data

test_x = np.array(test_data.iloc[:, 1:])

test_y = to_categorical(np.array(test_data.iloc[:, 0]))



# Reshape

train_x = np.reshape(train_x, (len(train_x), img_rows, img_cols, 1))

test_x = np.reshape(test_x, (len(test_x), img_rows, img_cols, 1))



# Scale to 0 - 1

train_x = train_x.astype('float32') / 255.

test_x = test_x.astype('float32') / 255.



print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

print(test_y.shape)
from keras.models import Model

from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten

from keras.optimizers import Adam



inputs = Input(shape=(28, 28, 1))



conv_layer = ZeroPadding2D(padding=(2,2))(inputs)



conv_layer = Conv2D(16, (5, 5), strides=(1,1), activation='relu')(conv_layer)

conv_layer = MaxPooling2D((2, 2))(conv_layer)



conv_layer = Conv2D(32, (3, 3), strides=(1,1), activation='relu')(conv_layer)

conv_layer = Conv2D(32, (3, 3), strides=(1,1), activation='relu')(conv_layer)

conv_layer = MaxPooling2D((2, 2))(conv_layer)



conv_layer = Conv2D(64, (3, 3), strides=(1,1), activation='relu')(conv_layer)



flatten = Flatten()(conv_layer)



fc_layer = Dense(256, activation='relu')(flatten)

fc_layer = Dense(64, activation='relu')(fc_layer)

outputs = Dense(10, activation='softmax')(fc_layer)



model = Model(inputs=inputs, outputs=outputs)



print(model.summary())
adam = Adam(lr=0.001)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=256, epochs=19, verbose=1, validation_data=(test_x, test_y))

model.save_weights('weights.h5')
score = model.evaluate(test_x, test_y, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])