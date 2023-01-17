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
def epocher(batch_size):
    # Create the print weights callback.
    from keras.callbacks import LambdaCallback
    history = []
    print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: history.append(clf.layers[0].get_weights()))

    # Build the model.
    clf = Sequential()
    clf.add(Dense(9, activation='relu', input_dim=3))
    clf.add(Dense(9, activation='relu'))
    clf.add(Dense(3, activation='softmax'))
    clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

    # Perform training.
    clf.fit(X_train, y_train, epochs=100, batch_size=batch_size, callbacks=[print_weights])
    
    # Return
    return history, clf
history, clf = epocher(batch_size=1000)
import matplotlib.pyplot as plt

plt.plot([history[n][0][0][0] for n in range(100)], range(100))
history, clf = epocher(batch_size=10)
plt.plot([history[n][0][0][0] for n in range(100)], range(100))
# history, clf = epocher(batch_size=1)