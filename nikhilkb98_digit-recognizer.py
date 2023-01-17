import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

from keras.utils import to_categorical

from keras.optimizers import Adam

from keras import Sequential

from sklearn.model_selection import train_test_split as tts

from keras.layers import Dense, Dropout, Flatten, Conv2D, AveragePooling2D

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.preprocessing.image import ImageDataGenerator

import warnings

from tensorflow.python.client import device_lib

import os

list = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        list.append(os.path.join(dirname, filename))
data = pd.read_csv(list[0])
data.shape
data.info()
data.head()
sns.countplot(data['label'])
X = np.asarray(data.iloc[:, 1:])

Y = np.asarray(data.iloc[:, 0])
print(X.shape, "  ", Y.shape)
X = X.reshape((X.shape[0], 28, 28))

Y = np.expand_dims(Y, axis = 1)
X.shape
Y.shape
X = X/X.max()


x = np.zeros((X.shape[0], 32, 32))



for i in range(0, X.shape[0]):

    x[i, :] = cv2.copyMakeBorder(X[i, :], 2, 2, 2, 2, cv2.BORDER_CONSTANT, value = 0)

    
x = np.expand_dims(x, axis = 3)

print(x.shape)

plt.imshow(x[3][:, :, 0])
y = to_categorical(Y, num_classes=10)
y.shape
x.shape
print(device_lib.list_local_devices())

warnings.filterwarnings('ignore')
model = Sequential()



model.add(Conv2D(filters = 6, kernel_size = (3, 3), activation = 'relu', input_shape = (32, 32, 1)))

model.add(AveragePooling2D())

model.add(Dropout(0.25))



model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu'))

model.add(AveragePooling2D())

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(units = 120, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(units = 84, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(units = 10, activation = 'softmax'))
optimizer = Adam(learning_rate = 0.001, beta_1 = 0.95, beta_2 = 0.99, amsgrad = True)
model.compile(optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
learning_rate = ReduceLROnPlateau(monitor = 'val_loss', factor=0.5, patience=5, verbose=1, mode='auto', min_lr=0.00001)

# early = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto')
history = model.fit(x, y, batch_size = 32, epochs = 30, verbose = 1, validation_split = 0.2, shuffle = True, callbacks = [learning_rate])
axes = plt.gca()



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('MODEL ACCURACY')

plt.ylabel('ACCURACY')

plt.xlabel('EPOCH')

plt.legend(['TRAIN', 'VALIDATE'], loc='bottom left')

axes.set_xlim([0, 30])

axes.set_ylim([0.7,1.0])

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('MODEL LOSS')

plt.ylabel('LOSS')

plt.xlabel('EPOCH')

plt.legend(['TRAIN', 'VALIDATE'], loc='upper right')

axes.set_xlim([0, 30])

axes.set_ylim([0.0,0.5])

plt.show()
x_train, x_val, y_train, y_val = tts(x, y, train_size = 0.9, random_state = 42)
data_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=10.0, zoom_range=0.1, fill_mode='nearest')

data_gen.fit(x_train)
model_gen = Sequential()



model_gen.add(Conv2D(filters = 6, kernel_size = (3, 3), activation = 'relu', input_shape = (32, 32, 1)))

model_gen.add(AveragePooling2D())

# model_gen.add(Dropout(0.25))



model_gen.add(Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu'))

model_gen.add(AveragePooling2D())

# model_gen.add(Dropout(0.25))



model_gen.add(Flatten())

model_gen.add(Dense(units = 120, activation = 'relu'))

model_gen.add(Dropout(0.25))

model_gen.add(Dense(units = 84, activation = 'relu'))

model_gen.add(Dropout(0.25))

model_gen.add(Dense(units = 10, activation = 'softmax'))



optimizer_gen = Adam(learning_rate = 0.01, beta_1 = 0.95, beta_2 = 0.99, amsgrad = True)

model_gen.compile(optimizer_gen, loss = 'categorical_crossentropy', metrics = ['accuracy'])
y_train.shape
history_gen = model_gen.fit_generator(data_gen.flow(x_train, y_train, 64), epochs = 30, verbose = 1, validation_data = (x_val, y_val), steps_per_epoch = x_train.shape[0] // 64, callbacks = [learning_rate])
axes = plt.gca()



plt.plot(history_gen.history['accuracy'])

plt.plot(history_gen.history['val_accuracy'])

plt.title('MODEL ACCURACY')

plt.ylabel('ACCURACY')

plt.xlabel('EPOCH')

plt.legend(['TRAIN', 'VALIDATE'], loc='bottom right')

axes.set_xlim([0, 30])

axes.set_ylim([0.5,1.0])

plt.show()



plt.plot(history_gen.history['loss'])

plt.plot(history_gen.history['val_loss'])

plt.title('MODEL LOSS')

plt.ylabel('LOSS')

plt.xlabel('EPOCH')

plt.legend(['TRAIN', 'VALIDATE'], loc='upper right')

axes.set_xlim([0, 30])

axes.set_ylim([0.0, 0.5])

plt.show()
data_test = pd.read_csv(list[1])

data_test.head()
X_test = np.asarray(data_test)

X_test = X_test.reshape((X_test.shape[0], 28, 28))

X_test = X_test/X_test.max()

x_test = np.zeros((X_test.shape[0], 32, 32))

for i in range(0, X_test.shape[0]):

    x_test[i, :] = cv2.copyMakeBorder(X_test[i, :], 2, 2, 2, 2, cv2.BORDER_CONSTANT, value = 0)

x_test = np.expand_dims(x_test, axis = 3)
from sklearn.metrics import confusion_matrix
y_pred = model_gen.predict_classes(x_val)
con_mat = confusion_matrix(np.argmax(y_val, axis = 1), y_pred)

con_mat
answers = model_gen.predict(x_test)

answers = np.argmax(answers, axis = 1)

print(answers)
data_final = {'ImageId': np.arange(1, x_test.shape[0]+1), 'Label': answers}

data_final
submit = pd.DataFrame(data = data_final)

submit
submit.to_csv('Digit_recognizer_nkb.csv', index = False)