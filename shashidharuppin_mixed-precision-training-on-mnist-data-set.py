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
!nvidia-smi -L
import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.mixed_precision import experimental as mixed_precision



policy = mixed_precision.Policy('mixed_float16')

mixed_precision.set_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)

print('Variable dtype: %s' % policy.variable_dtype)
inputs = keras.Input(shape=(784,), name='digits')

if tf.config.list_physical_devices('GPU'):

    print('The model will run with 4096 units on a GPU')

    num_units = 4096

else:

  # Use fewer units on CPUs so the model finishes in a reasonable amount of time

    print('The model will run with 64 units on a CPU')

    num_units = 64



dense1 = layers.Dense(num_units, activation='relu', name='dense_1')

x = dense1(inputs)

dense2 = layers.Dense(num_units, activation='relu', name='dense_2')

x = dense2(x)
print('x.dtype: %s' % x.dtype.name)

# 'kernel' is dense1's variable

print('dense1.kernel.dtype: %s' % dense1.kernel.dtype.name)
# CORRECT: softmax and model output are float32

x = layers.Dense(10, name='dense_logits')(x)

outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x)

print('Outputs dtype: %s' % outputs.dtype.name)
outputs = layers.Activation('linear', dtype='float32')(outputs)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss='sparse_categorical_crossentropy',

              optimizer=keras.optimizers.RMSprop(),

              metrics=['accuracy'])



(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 255

x_test = x_test.reshape(10000, 784).astype('float32') / 255
initial_weights = model.get_weights()
history = model.fit(x_train, y_train,

                    batch_size=8192,

                    epochs=5,

                    validation_split=0.2)



test_scores = model.evaluate(x_test, y_test, verbose=2)

print('Test loss:', test_scores[0])

print('Test accuracy:', test_scores[1])
loss_scale = policy.loss_scale

print('Loss scale: %s' % loss_scale)