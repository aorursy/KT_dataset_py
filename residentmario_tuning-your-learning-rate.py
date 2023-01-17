##############################
# MODEL BUILDING BOILERPLATE #
##############################

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
import pandas as pd

X_train = np.random.random((1000, 3))
y_train = pd.get_dummies(np.argmax(X_train[:, :3], axis=1)).values
X_test = np.random.random((100, 3))
y_test = pd.get_dummies(np.argmax(X_test[:, :3], axis=1)).values
import numpy as np
from keras.callbacks import LearningRateScheduler

###########################################
# DEFINE A STEPPED LEARNING RATE SCHEDULE #
###########################################

lr_sched = LearningRateScheduler(lambda epoch: 1e-4 * (0.75 ** np.floor(epoch / 2)))

# Build the model.
clf = Sequential()
clf.add(Dense(9, activation='relu', input_dim=3))
clf.add(Dense(9, activation='relu'))
clf.add(Dense(3, activation='softmax'))
clf.compile(loss='categorical_crossentropy', optimizer=SGD())

# Perform training.
clf.fit(X_train, y_train, epochs=10, batch_size=500, callbacks=[lr_sched])