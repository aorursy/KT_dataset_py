import numpy as np 

import matplotlib.pyplot as plt

from keras.models import Sequential 

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import np_utils

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_val, y_val = X_train[50000:60000], y_train[50000:60000]

X_train, y_train = X_train[:50000], y_train[:50000]

X_train.shape
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train.shape
Y_train = np_utils.to_categorical(y_train, 10)

Y_val = np_utils.to_categorical(y_val, 10)

Y_test = np_utils.to_categorical(y_test, 10)

print("Dữ liệu y ban đầu: ", y_train[0])

print("Dữ liệu y sau one-hot encoding: ", Y_train[0])
model = Sequential()

model.add(Conv2D(32, (3, 3), activation = 'sigmoid', input_shape = (28,28,1)))

model.add(Conv2D(32, (3, 3), activation = 'sigmoid'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(128, activation = 'sigmoid'))

model.add(Dense(10, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy',

             optimizer = 'adam',

             metrics = ['accuracy'])
H = model.fit(X_train, Y_train, validation_data = (X_val, Y_val),

             batch_size = 32, epochs = 10, verbose = 1)
fig = plt.figure(figsize = (12, 6))

numOfEpoch = 10

plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label = 'training loss')

plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label = 'validation loss')

plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label = 'accuracy')

plt.plot(np.arange(0, numOfEpoch), H.history['val_accuracy'], label = 'validation accuracy')

plt.title('Accuracy and Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss|Accuracy')

plt.legend()
score = model.evaluate(X_test, Y_test, verbose = 1)

score
plt.imshow(X_test[0].reshape(28, 28), cmap = 'gray')

y_predict = model.predict(X_test[0].reshape(1,28,28,1))



print('Giá trị dự đoán: ', np.argmax(y_predict))