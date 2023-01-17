#

# Generate dummy data.

#



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
#

# Build a KerasClassifier wrapper object.

# I had trouble getting the callable class approach to work. The method approach seems to be pretty universial anyway.

#



from keras.wrappers.scikit_learn import KerasClassifier



# Doesn't work?

# class TwoLayerFeedForward:

#     def __call__():

#         clf = Sequential()

#         clf.add(Dense(9, activation='relu', input_dim=3))

#         clf.add(Dense(9, activation='relu'))

#         clf.add(Dense(3, activation='softmax'))

#         clf.compile(loss='categorical_crossentropy', optimizer=SGD())

#         return clf



def twoLayerFeedForward():

    clf = Sequential()

    clf.add(Dense(9, activation='relu', input_dim=3))

    clf.add(Dense(9, activation='relu'))

    clf.add(Dense(3, activation='softmax'))

    clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])

    return clf





# clf = KerasClassifier(TwoLayerFeedForward(), epochs=100, batch_size=500, verbose=0)

clf = KerasClassifier(twoLayerFeedForward, epochs=100, batch_size=500, verbose=0)
from sklearn.model_selection import StratifiedKFold, cross_val_score



trans = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)



import pandas as pd



# Keras classifiers work with one hot encoded categorical columns (e.g. [[1 0 0], [0 1 0], ...]).

# StratifiedKFold works with categorical encoded columns (e.g. [1 2 3 1 ...]).

# This requires juggling the representation at shuffle time versus at runtime.

scores = []

for train_idx, test_idx in trans.split(X_train, y_train.argmax(axis=1)):

    X_cv, y_cv = X_train[train_idx], pd.get_dummies(y_train.argmax(axis=1)[train_idx]).values

    clf.fit(X_cv, y_cv)

    scores.append(clf.score(X_cv, y_cv))
scores