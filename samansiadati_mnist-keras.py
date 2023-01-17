from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
y_train.shape
import matplotlib.pyplot as plt
plt.imshow(X_train[0])
plt.imshow(X_train[1])
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i])
    plt.title("Class {}".format(y_train[i]))
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)
from keras.utils import np_utils
import numpy as np
classes = 10
Y_train = np_utils.to_categorical(y_train, classes)
Y_test = np_utils.to_categorical(y_test, classes)
Y_train.shape
Y_test.shape
Y_train.shape[1]
from keras.models import Sequential
model_krs = Sequential()
from keras import layers
from keras.layers.core import Dropout
model_krs.add(layers.Dense(512, input_shape=(784,), activation='relu'))
              
#### Dropout for not memorize or overfitting the train data
model_krs.add(Dropout(0.2)) 
model_krs.add(layers.Dense(512, activation='relu'))
model_krs.add(Dropout(0.2)) 
model_krs.add(layers.Dense(10, activation='softmax'))
model_krs.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_krs.summary()
history = model_krs.fit(X_train, Y_train,
          epochs=10,
          verbose=1,
          validation_data=(X_test, Y_test),
          batch_size=128)
accuracy = model_krs.evaluate(X_train, Y_train, verbose=False)
print("Training Score: {:.4f}".format(accuracy[0]))
print("Training Accuracy: {:.4f}".format(accuracy[1]))
accuracy = model_krs.evaluate(X_test, Y_test, verbose=False)
print("Testing Accuracy: {:.4f}".format(accuracy[0]))
print("Testing Accuracy: {:.4f}".format(accuracy[1]))
Predict = model_krs.predict_classes(X_test)
Right_predict = np.nonzero(Predict == y_test)[0]
Wrong_predict = np.nonzero(Predict != y_test)[0]
plt.figure()
for i, correct in enumerate(Right_predict[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28))
    plt.title("Predicted {}, Class {}".format(Predict[correct], y_test[correct]))
plt.figure()
for i, incorrect in enumerate(Wrong_predict[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28))
    plt.title("Predicted {}, Class {}".format(Predict[incorrect], y_test[incorrect]))