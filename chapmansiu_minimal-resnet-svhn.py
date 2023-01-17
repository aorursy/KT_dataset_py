import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, load_digits

import  tensorflow  as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Reshape
from keras.models import Model
from keras.layers import Conv1D, Conv2D, Conv3D, SeparableConv1D, MaxPooling1D, MaxPooling2D

from keras import backend as K
from keras import initializers, layers
from keras.engine.topology import Layer

from keras.utils import to_categorical

from sklearn import metrics
from sklearn import tree, ensemble
import scipy.io as sio
from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)
try:
    from keras_applications.resnet50 import identity_block, conv_block
except:
    from keras.applications.resnet50 import identity_block, conv_block
from keras import utils as keras_utils
train_dat = sio.loadmat("../input/train_32x32.mat")

n_samples = train_dat['X'].T.shape[0]
X = train_dat['X'].T
X = np.swapaxes(X, 1, 3)
y_ = train_dat['y'].flatten()
y = to_categorical(y_)

test_dat = sio.loadmat("../input/test_32x32.mat")
#n_samples = test_dat['X'].T.shape[0]
X_test = test_dat['X'].T
X_test = np.swapaxes(X_test, 1, 3)
y_test_ = test_dat['y'].flatten()
y_test = to_categorical(y_test_)

nepochs=5
nclasses = y.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# using resnet50 as benchmark
#from keras.applications.resnet50 import ResNet50

main_input = Input(shape=(32, 32, 3), name='main_input')
# replace with resnet50 if you wish
#model = ResNet(include_top=True, weights=None, input_tensor=main_input, input_shape=(32, 32, 3), pooling='max', classes=nclasses)
x = layers.Conv2D(16, (3, 3),
                  strides=(2, 2),
                  padding='same',
                  kernel_initializer='he_normal',
                  name='conv1')(main_input)
x = conv_block(x, 3, [16, 16, 32], stage=2, block='a', strides=(1, 1))
x = identity_block(x, 3, [16, 16, 32], stage=2, block='c')
x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
x = layers.Dense(nclasses, activation='softmax', name='fc1000')(x)


model = Model(main_input, x)

model.compile(optimizer='adam', 
            loss='categorical_crossentropy',
            metrics=['accuracy'])

model.summary()

hist = model.fit(X, y, validation_data=(X_test, y_test), epochs=5, verbose=1)
print(pd.DataFrame(hist.history).iloc[-1])


model.evaluate([X_test], [y_test])