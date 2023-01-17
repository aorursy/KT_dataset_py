from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
%matplotlib inline 
data_train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
data_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

x_train = np.array(data_train.iloc[:, 1:])
y_train = np.array(data_train.iloc[:, 0])

x_test = np.array(data_test.iloc[:, 1:])
y_test = np.array(data_test.iloc[:, 0])
object_classes = ['футболка', 'штаны', 'кофта', 'платье', 'куртка', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']
plt.figure(figsize=(10, 10))
for i in range(0, 20):
    plt.subplot(5, 10, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i].reshape(28,28), cmap='gray')
    plt.xlabel(object_classes[y_train[i]])
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_train = x_train / 255 

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32')
x_test = x_test / 255 
y_train = utils.to_categorical(y_train)

# y_test = utils.to_categorical(y_test)
model = Sequential()
model.add(Conv2D(32, (3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=200, epochs=50, verbose=1)
model.save('fashion-mnist.h5')
from tensorflow.keras.models import load_model

plt.figure(figsize=(10, 10))
plt.subplot(5, 10, 1)
plt.xticks([])
plt.yticks([])
plt.imshow(x_test[0].reshape(28,28), cmap='gray')
prediction = model.predict(x_test)
prediction = np.argmax(prediction[0])
print('Номер:', prediction)
print('Название:', object_classes[prediction])