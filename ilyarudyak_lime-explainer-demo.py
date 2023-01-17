!pip install aix360
!pip install keras==2.3.0
from __future__ import print_function

import warnings

# Supress jupyter warnings if required for cleaner output

warnings.simplefilter('ignore')



import numpy as np

import pandas as pd



import keras

import keras.layers



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential # Sequeantial layer addition



from aix360.algorithms.lime import LimeImageExplainer



print('Using keras:', keras.__version__)
# Load dataset

from keras.datasets import mnist

# Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test).

(train, train_labels), (test, test_labels) = mnist.load_data()



# save input image dimensions

img_rows = train.shape[1]

img_cols = train.shape[2]



# Get classes and number of values

value_counts = pd.value_counts(train_labels).sort_index()

num_classes = value_counts.count()



train = train/255

test = test/255



train = train.reshape(train.shape[0], img_rows, img_cols, 1)

test = test.reshape(test.shape[0], img_rows, img_cols, 1)
def to_rgb(x):

    x_rgb = np.zeros((x.shape[0], 28, 28, 3))

    for i in range(3):

        x_rgb[..., i] = x[..., 0]

    return x_rgb



train_rgb = to_rgb(train)

test_rgb = to_rgb(test)
train_rgb.shape, test_rgb.shape
train_rgb[20, 20, 20, :], train[20, 20, 20, :]
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(img_rows, img_cols, 3)),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(num_classes)

])



model.compile(loss='sparse_categorical_crossentropy',

      optimizer='adam',

      metrics=['accuracy'])



batch_size = 128

epochs = 3



model.fit(train_rgb, train_labels,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(test_rgb, test_labels))



score = model.evaluate(test_rgb, test_labels, verbose=0)



print('Test loss:', score[0])

print('Test accuracy:', score[1])

limeExplainer = LimeImageExplainer()
limeExplainer.explain_instance(test_rgb[0], model.predict_proba)