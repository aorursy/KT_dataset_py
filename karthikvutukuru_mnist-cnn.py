# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, Dense, Dropout

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

from tensorflow.keras.utils import to_categorical, plot_model

from tensorflow.keras.datasets import mnist



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load data

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Compute no. of labels

num_labels = len(np.unique(y_train))



# Convert labels to OneHotEncoding

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
# input shape

image_size = X_train.shape[1]

# Resize and Resample

X_train = np.reshape(X_train, [-1, image_size, image_size, 1])

X_test = np.reshape(X_test, [-1, image_size, image_size, 1] )



X_train = X_train.astype(np.float32)/255.

X_test = X_test.astype(np.float32)/255.

# Network parameters

input_shape = (image_size, image_size, 1) # 1 for Grayscale image

BATCH_SIZE = 128

KERNEL_SIZE = 3

POOL_SIZE = 2

FILTERS=64

DROPOUT = 0.2
# Model stack of CNN-RELU-MAXPOOL

model = Sequential()

model.add(Conv2D(filters=FILTERS, 

                 kernel_size=KERNEL_SIZE,

                 activation='relu',

                 input_shape=input_shape

                ))

model.add(MaxPooling2D(POOL_SIZE))

model.add(Conv2D(filters=FILTERS,

                kernel_size=KERNEL_SIZE,

                activation='relu'

                ))

model.add(MaxPooling2D(POOL_SIZE))

model.add(Conv2D(filters=FILTERS,

                kernel_size=KERNEL_SIZE,

                activation='relu'

                ))

model.add(Flatten())

# Dropout

model.add(Dropout(DROPOUT))

# Output layer

model.add(Dense(num_labels))

model.add(Activation('softmax'))

model.summary()

plot_model(model, to_file='/kaggle/working/cnn_model.png', show_shapes=True)
# Compile Model

model.compile(loss='categorical_crossentropy',

optimizer='adam',

metrics=['accuracy'])

# train the network

model.fit(X_train, y_train, epochs=10, batch_size=BATCH_SIZE)

_, acc = model.evaluate(X_test,

y_test,

batch_size=BATCH_SIZE,

verbose=0)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))