

import numpy as np

import pandas as pd



from keras.models import Sequential

from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten

from keras.optimizers import adam

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils.np_utils import to_categorical