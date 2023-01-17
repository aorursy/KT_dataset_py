import numpy as np

import pandas as pd

import os

import pickle

import matplotlib.pyplot as plt

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
DATA_DIR = '../input/cifar10-preprocessed/data.pickle'

LABELS_DIR = '../input/cifar10-preprocessed/labels.txt'

EPOCHS = 400

BATCH_SIZE = 64
with open(DATA_DIR, 'rb') as f:

    data = pickle.load(f)

    

X_train = data['x_train'].copy()

y_train = data['y_train'].copy()

X_val = data['x_validation'].copy()

y_val = data['y_validation'].copy()

X_test = data['x_train'].copy()

y_test = data['y_train'].copy()



del(data)
labels = []



with open(LABELS_DIR, 'r') as f:

    for label in f:

        label = label.strip()

        labels.append(label)

        

labels
print(f'''

X_train: {X_train.shape}

y_train: {y_train.shape},

X_val: {X_val.shape},

y_val: {y_val.shape},

X_test: {X_test.shape},

y_test: {y_test.shape}

''')
def proceed_X(data):

    data = np.transpose(data, (0, 2, 3, 1))

    data = data.astype('float32')

    return data
X_train = proceed_X(X_train)

X_val = proceed_X(X_val)

X_test = proceed_X(X_test)
def proceed_y(data):

    data = to_categorical(data, dtype='int')

    return data
y_train = proceed_y(y_train)

y_val = proceed_y(y_val)

y_test = proceed_y(y_test)
for i in range(5):

    plt.imshow(X_train[i])

    plt.show()
model = Sequential()



# 32x32x3

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu', input_shape=(32, 32, 3)))

model.add(BatchNormalization())

# 32x32x3

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))

model.add(BatchNormalization())

# 16x16x3

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.2))



# 16x16x3

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))

model.add(BatchNormalization())

# 16x16x3

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))

model.add(BatchNormalization())

# 8x8x3

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.2))



# 8x8x3

model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))

model.add(BatchNormalization())

# 8x8x3

model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))

model.add(BatchNormalization())

# 4x4x3

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.2))



model.add(Flatten())



model.add(Dense(2048, activation='relu', kernel_initializer='he_uniform'))

model.add(Dropout(0.5))



model.add(Dense(2048, activation='relu', kernel_initializer='he_uniform'))

model.add(Dropout(0.5))



model.add(Dense(10, activation='softmax'))



opt = SGD(lr=1e-3, momentum=0.9)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))
# Vizualize history

# fig, ax = plt.subplots(2,1)



# ax[0].plot(history.history['loss'], color='b', label="Training loss")

# ax[0].plot(history.history['val_loss'], color='r', label="Validation loss")

# legend = ax[0].legend(loc='best', shadow=True)



# ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

# ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")

# legend = ax[1].legend(loc='best', shadow=True)
datagen = ImageDataGenerator(

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True)
history = model.fit_generator(

    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),

    steps_per_epoch=X_train.shape[0] // BATCH_SIZE,

    epochs=EPOCHS,

    validation_data=(X_val, y_val),

    verbose=1)
# Vizualize history

fig, ax = plt.subplots(2,1)



ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="Validation loss")

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
loss, acc = model.evaluate(X_test, y_test)
print(f'acc: {acc} loss: {loss}')
model.save('final_model.h5')