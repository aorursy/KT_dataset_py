# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import random
from keras.models import Sequential

from keras import layers

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import ModelCheckpoint

from keras.models import load_model

# from keras.callbacks import TensorBoard

from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# import seaborn as sns

# sns.set()
train_df = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test_df = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

val_df = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
train_df.tail()
val_df.tail()
test_df.tail()
train_df = pd.concat([train_df, val_df], axis = 0)
Y_train = train_df['label']

Y_id = test_df['id']
train_df.drop('label', axis = 1, inplace = True)

test_df.drop('id', axis = 1, inplace = True)
# X_train = train_df.to_numpy()

# X_test = test_df.to_numpy()
print(train_df.shape)

print(test_df.shape)
X_train = train_df.values.reshape(train_df.shape[0], 28, 28, 1)

X_test = test_df.values.reshape(test_df.shape[0], 28, 28, 1)
Y_train = np.asarray(Y_train)

print(Y_train.shape)
print(X_train.shape)

print(X_test.shape)
def plot_digit(dataset, samples, labels):

#     Generate a random set of samples and plot them

#     random_samples = []

    for pts in range(samples):

        r_n = random.randint(0,dataset.shape[0])

#         random_samples  = [23,43,445,44,34]

        img = dataset[r_n].reshape(28,28)

        plt.imshow(img, cmap = 'gray')

        plt.title(labels[r_n])

        plt.show()

        
# plot_digit(X_train, 5, Y_train)
Y_train = to_categorical(Y_train, num_classes = 10)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, stratify = Y_train, random_state = 31, shuffle = True, test_size = 0.2)
train_datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.3,

                                   height_shift_range=0.3, brightness_range=None, shear_range=0.3,

                                   zoom_range=0.3, horizontal_flip=False,

                                   vertical_flip=False, rescale=1/255.,

                                   data_format='channels_last', validation_split=0.0,

                                   interpolation_order=1, dtype='float32')
X_val = X_val / 255.                    # Scaling validation Data
X_test = X_test / 255.
print(X_train.shape)

print(X_val.shape)

print(Y_train.shape)

print(Y_val.shape)
train_datagen.fit(X_train)
model = Sequential()



model.add(layers.Conv2D(128, 3, input_shape = (28,28,1), padding = 'same', activation = None))

model.add(layers.LeakyReLU(alpha = 0.3))

model.add(layers.BatchNormalization(axis=1))

model.add(layers.Dropout(0.2))



model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))



model.add(layers.Conv2D(64, 3, padding = 'same', activation = None))

model.add(layers.LeakyReLU(alpha = 0.3))

model.add(layers.BatchNormalization(axis=1))

model.add(layers.Dropout(0.3))



model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))



model.add(layers.Conv2D(64, 3, padding = 'same', activation = None))

model.add(layers.LeakyReLU(alpha = 0.3))

model.add(layers.BatchNormalization(axis=1))

model.add(layers.Dropout(0.3))



model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))



model.add(layers.Conv2D(64, 3, padding = 'same', activation = None))

model.add(layers.LeakyReLU(alpha = 0.3))

model.add(layers.BatchNormalization(axis=1))

model.add(layers.Dropout(0.3))



model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))



model.add(layers.Conv2D(32, 3, padding = 'same', activation = None))

model.add(layers.LeakyReLU(alpha = 0.3))

model.add(layers.BatchNormalization(axis=1))

model.add(layers.Dropout(0.3))



model.add(layers.Flatten())

model.add(layers.Dense(256))

model.add(layers.LeakyReLU(alpha = 0.3))

model.add(layers.Dense(10, activation = 'softmax'))



model.summary()
opt1 = optimizers.Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999)

opt2 = optimizers.Nadam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999)
filepath1 = 'kannada_cnn_model1.hdf5'

filepath2 = 'kannada_cnn_model2.hdf5'

reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

checkpoint1 = ModelCheckpoint(filepath1, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

checkpoint2 = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

callback_l = [reducelr, checkpoint1]

callback_l2 = [reducelr, checkpoint2]
model.compile(optimizer = opt1, loss = 'categorical_crossentropy', metrics = ['acc'])
epochs = 50

batch_size = 128
# print(Y_train[1])
history = model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size = batch_size ),

                              epochs = epochs, validation_data = (X_val, Y_val), verbose = 1,

                              steps_per_epoch=X_train.shape[0] // batch_size, callbacks = callback_l)
# acc = history.history['acc']

# val_acc = history.history['val_acc']

# loss = history.history['loss']

# val_loss = history.history['val_loss']



# epochs = range(1, len(acc) + 1)



# plt.plot(epochs, acc, 'g', label='Training acc')

# plt.plot(epochs, val_acc, 'b', label='Validation acc')

# plt.title('Training and validation accuracy')

# plt.legend()

# plt.figure()



# plt.plot(epochs, loss, 'g', label='Training loss')

# plt.plot(epochs, val_loss, 'b', label='Validation loss')

# plt.title('Training and validation loss')

# plt.legend()

# plt.show()
# model.compile(optimizer = opt2, loss = 'categorical_crossentropy', metrics = ['acc'])
# history = model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size = batch_size ),

#                               epochs = 50, validation_data = (X_val, Y_val), verbose = 1,

#                               steps_per_epoch=X_train.shape[0] // batch_size, callbacks = callback_l2)
# acc = history.history['acc']

# val_acc = history.history['val_acc']

# loss = history.history['loss']

# val_loss = history.history['val_loss']



# epochs = range(1, len(acc) + 1)



# plt.plot(epochs, acc, 'g', label='Training acc')

# plt.plot(epochs, val_acc, 'b', label='Validation acc')

# plt.title('Training and validation accuracy')

# plt.legend()

# plt.figure()



# plt.plot(epochs, loss, 'g', label='Training loss')

# plt.plot(epochs, val_loss, 'b', label='Validation loss')

# plt.title('Training and validation loss')

# plt.legend()

# plt.show()
best_model = Sequential()
best_model = load_model(filepath1)
y_pred = best_model.predict(X_test)

y_pred = np.argmax(y_pred, axis = 1)
sample_sub = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

sample_sub['label'] = y_pred

sample_sub.to_csv('submission.csv', index = False)
sample_sub.tail()