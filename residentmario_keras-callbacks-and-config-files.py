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


# Reusable model fit wrapper.
def epocher(batch_size=500, epochs=10, callbacks=None):
    # Build the model.
    clf = Sequential()
    clf.add(Dense(9, activation='relu', input_dim=3))
    clf.add(Dense(9, activation='relu'))
    clf.add(Dense(3, activation='softmax'))
    clf.compile(loss='categorical_crossentropy', optimizer=SGD())

    # Perform training.
    clf.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[callbacks])
    return clf
from keras.callbacks import LambdaCallback

history = []
weight_history = LambdaCallback(on_epoch_end=lambda batch, logs: history.append((batch, logs)))
clf = epocher(callbacks=weight_history)
clf.history.history
from keras.callbacks import Callback

class WeightHistory(Callback):
    def __init__(self):
        self.tape = []
        
    def on_epoch_end(self, batch, logs={}):
        self.tape.append(self.model.get_weights()[0])

wh = WeightHistory()
clf = epocher(callbacks=wh)
wh.tape[0]
keras.callbacks.TerminateOnNaN
clf.get_config()
my_config = clf.get_config()
Sequential.from_config(my_config)