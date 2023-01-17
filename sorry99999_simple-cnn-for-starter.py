import numpy as np
import pandas as pd

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

batch_size = 64
num_classes = 10
epochs = 20
input_shape = (28, 28, 1)
train = pd.read_csv('../input/train.csv')
print(train.shape)
train.head()

test = pd.read_csv('../input/test.csv')
print(test.shape)
test.head()
X = train.iloc[:, 1:].values.astype('float32')
y = train.iloc[:, 0].values.astype('int32')

X_test = test.values.astype('float32')
X = X / 255.0
X_test = X_test / 255.0
X = X.reshape(X.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

y = keras.utils.to_categorical(y, num_classes)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', input_shape=input_shape))
model.add(Dropout(0.5))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])
model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
y_test = np.argmax(model.predict(X_test), axis=1)
submission = pd.DataFrame({'ImageId': list(range(1, len(y_test) + 1)),
                           'Label': y_test})
submission.to_csv('final.csv', index=False, header=True)