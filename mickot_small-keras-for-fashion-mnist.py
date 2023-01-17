import pandas as pd
import numpy as np
np.random.seed(123)
df_train = pd.read_csv('../input/fashion-mnist_train.csv')
df_test = pd.read_csv('../input/fashion-mnist_test.csv')
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
batch_size = 64
nb_classes = 10

img_rows = 28
img_cols = 28

X_train = df_train[df_train.columns[1:]].values
y_train = df_train[df_train.columns[0]].values
X_test = df_test[df_test.columns[1:]].values
y_test = df_test[df_test.columns[0]].values
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
import matplotlib.pyplot as plt
%matplotlib inline
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i, 0], cmap = 'gray')
    plt.axis('off')
model = Sequential()

model.add(Convolution2D(6, 5, 5, input_shape = (1, img_rows, img_cols), border_mode = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), dim_ordering="th"))
model.add(Convolution2D(16, 5, 5, border_mode = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), dim_ordering="th"))
model.add(Convolution2D(120, 5, 5, dim_ordering="th"))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])
epochs = 10

hist = model.fit(X_train,y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (X_test, y_test))
print('Quick visualization of model training history')
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
