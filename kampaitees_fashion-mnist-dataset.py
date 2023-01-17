import numpy as np

import matplotlib.pyplot as plt



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from keras.utils.np_utils import to_categorical

from keras.preprocessing import image

from keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train_P = X_train.reshape(60000, 28, 28, -1)
X_train_P = X_train_P / 255.0
X_test_P = X_test.reshape(10000, 28, 28, 1)

X_test_P = X_test_P / 255.0
Y_train = to_categorical(y_train, 10)

Y_test = to_categorical(y_test, 10)
model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))

model.add(MaxPooling2D((2, 2)))



model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))

model.add(MaxPooling2D((2, 2)))



# model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))

# model.add(MaxPooling2D((2, 2)))



# model.add(Conv2D(filters = 512, kernel_size = (3, 3), activation = 'relu'))

# model.add(MaxPooling2D((2, 2)))



model.add(Flatten())



model.add(Dense(256, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(128, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(64, activation = 'relu'))



model.add(Dense(10, activation = 'softmax'))



model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train_P, Y_train, batch_size = 32, epochs = 10, validation_data = (X_test_P, Y_test))
model.summary()
y_predictions = model.predict_classes(X_test_P)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_predictions))