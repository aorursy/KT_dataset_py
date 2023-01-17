import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
# Generate dummy data
import numpy as np
y_train = np.random.randint(2, size=(1000, 1))
X_train = y_train + (np.random.normal(size=(1000, 1)) / 5)
y_test = np.random.randint(2, size=(100, 1))
X_test = y_test + np.random.random(size=(100, 1))
import matplotlib.pyplot as plt
plt.scatter(X_train[:, 0], y_train, marker='x', c=y_train)
plt.axis('off')
clf = Sequential()
clf.add(Dense(2, activation='linear', input_shape=(1,), name='hidden'))
clf.add(Dense(1, activation='linear', name='out'))
clf.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=['accuracy'])

clf.fit(X_train, y_train, epochs=10, batch_size=128)
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(clf).create(prog='dot', format='svg'))
y_pred = clf.predict(X_test)
y_pred[:5]
y_pred = clf.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
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
clf.add(Dense(3, activation='linear', name='out'))
clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

clf.fit(X_train, y_train, epochs=10, batch_size=128)
clf.predict(X_test)[:5]
y_test_pred = np.zeros(shape=y_test.shape)

for x, y in enumerate(clf.predict(X_test).argmax(axis=1)):
    y_test_pred[x][y] = 1
f'Achieved accuracy score of {(y_test_pred == y_test).sum().sum() / (y_test.shape[0] * y_test.shape[1])}'
clf = Sequential()
clf.add(Dense(3, activation='linear', input_shape=(1,), name='hidden'))
clf.add(Dense(3, activation='softmax', name='out'))
clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

clf.fit(X_train, y_train, epochs=20, batch_size=128)
y_test_pred = clf.predict(X_test)
y_test_pred[:5]
y_test_pred.argmax(axis=1)
y_test_pred = np.zeros(shape=y_test.shape)

for x, y in enumerate(clf.predict(X_test).argmax(axis=1)):
    y_test_pred[x][y] = 1
f'Achieved accuracy score of {(y_test_pred == y_test).sum().sum() / (y_test.shape[0] * y_test.shape[1])}'