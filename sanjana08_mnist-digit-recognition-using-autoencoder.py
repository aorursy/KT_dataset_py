import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from keras.models import Model
from keras.layers import Input, Dense
from keras.datasets import mnist
train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")
test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")
train.shape, test.shape
train.head()
X_train = np.asarray(train.drop(['label'], axis=1))
y_train = np.asarray(train['label'])
X_train.shape, y_train.shape
X_test = np.asarray(test.drop(['label'], axis=1))
y_test = np.asarray(test['label'])
X_test.shape, y_test.shape
X_train = X_train/255
X_test = X_test/255
print("y_train values", np.unique(y_train))
print("y_test values", np.unique(y_test))
unique, counts = np.unique(y_train, return_counts=True)
print("y_train distribuion", dict(zip(unique, counts)))
plt.hist(y_train, ec='black')
unique, counts = np.unique(y_test, return_counts=True)
print("y_test distribuion", dict(zip(unique, counts)))
plt.hist(y_test, ec='black')
rows = 4
cols = 4
f = plt.figure(figsize=(rows+5,cols+5))
for i in range(rows*cols):
    f.add_subplot(rows,cols,i+1)
    plt.imshow(X_train[i].reshape([28,28]), cmap='gray')
input_size =784
h1_size = 196
h2_size = 32
x = Input(shape=(input_size,))

h1 = Dense(h1_size, activation='relu')(x)
h = Dense(h2_size, activation='relu')(h1)
h2 = Dense(h1_size, activation='relu')(h)

o = Dense(input_size, activation='sigmoid')(h2)
autoencoder = Model(x,o)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
history = autoencoder.fit(X_train, X_train, 
                          batch_size=50, 
                          epochs=50, 
                          shuffle=True, 
                          validation_data=(X_test, X_test))

decoded_digits = autoencoder.predict(X_test)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Autoencoder Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.plot(history.history['accuracy'])
n=10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i+50].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_digits[i+50].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()