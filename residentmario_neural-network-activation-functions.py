import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np

def sample_threeclass(n, ratio=0.8):
    np.random.seed(42)
    y_0 = np.random.randint(2, size=(n, 1))
    switch = (np.random.random(size=(n, 1)) <= ratio)
    y_1 = ~y_0 & switch
    y_2 = ~y_0 & ~switch
    y = np.concatenate([y_0, y_1, y_2], axis=1)
    
    X = y_0 + (np.random.normal(size=n) / 5)[np.newaxis].T
    return (X, y)


X_train, y_train = sample_threeclass(1000)
X_test, y_test = sample_threeclass(100)
clf = Sequential()
clf.add(Dense(3, activation='linear', input_shape=(1,), name='hidden'))
clf.add(Dense(3, activation='softmax', name='out'))
clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

clf.fit(X_train, y_train, epochs=10, batch_size=16)
def logistic_func(x): return np.e**x/(np.e**x + 1)

import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(np.arange(-5, 5, 0.2), [logistic_func(x) for x in np.arange(-5, 5, 0.2)])
plt.axis('off')
clf = Sequential()
clf.add(Dense(3, activation='sigmoid', input_shape=(1,), name='hidden'))
clf.add(Dense(3, activation='softmax', name='out'))
clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

clf.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
fig = plt.figure(figsize=(12, 8))
plt.plot(range(len(clf.history.history['acc'])), clf.history.history['acc'], linewidth=4)
import seaborn as sns; sns.despine()
plt.title("Sigmoid Activation Accuracy Per Epoch", fontsize=20)
pass
plt.plot(np.arange(-5, 5, 0.2), [np.tanh(x) for x in np.arange(-5, 5, 0.2)])
plt.axis('off')
clf = Sequential()
clf.add(Dense(3, activation='tanh', input_shape=(1,), name='hidden'))
clf.add(Dense(3, activation='softmax', name='out'))
clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

clf.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
fig = plt.figure(figsize=(12, 8))
plt.plot(range(len(clf.history.history['acc'])), clf.history.history['acc'], linewidth=4)
import seaborn as sns; sns.despine()
plt.title("Tanh Activation Accuracy Per Epoch", fontsize=20)
pass
def relu(x):
    return 0 if x <= 0 else x

plt.plot(np.arange(-5, 5, 0.2), [relu(x) for x in np.arange(-5, 5, 0.2)])
plt.axis('off')
pass
clf = Sequential()
clf.add(Dense(3, activation='relu', input_shape=(1,), name='hidden'))
clf.add(Dense(3, activation='softmax', name='out'))
clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

clf.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
fig = plt.figure(figsize=(12, 8))
plt.plot(range(len(clf.history.history['acc'])), clf.history.history['acc'], linewidth=4)
import seaborn as sns; sns.despine()
plt.title("ReLU Activation Accuracy Per Epoch", fontsize=20)
pass
def leaky_relu(x):
    return 0.01 if x <= 0 else x

plt.plot(np.arange(-5, 5, 0.2), [relu(x) for x in np.arange(-5, 5, 0.2)])
plt.axis('off')
pass