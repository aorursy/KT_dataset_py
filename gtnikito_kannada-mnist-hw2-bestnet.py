# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/Kannada-MNIST/train.csv")

train_data.shape
import matplotlib.pyplot as plt



show_exmpl = train_data.values[:8, :-1]

plt.figure(1, figsize=(14, 7))

for i in range(8):

    plt.subplot(2, 4, i + 1)

    plt.imshow(show_exmpl[i].reshape((28, 28)), cmap='gray')
X_train_test = train_data.values[:, 1:]

y_train_test = train_data.label.values



X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size = 0.02, random_state=42) 



print('Train shapes: ', X_train.shape, y_train.shape)

print('Test shapes: ', X_test.shape, y_test.shape)
print(np.min(X_train), np.max(X_train))

X_train_max = np.max(X_train)

X_train = X_train / (0.5 * X_train_max) - 1

print(np.min(X_train), np.max(X_train))



print(np.min(X_test), np.max(X_test))

X_test = X_test / (0.5 * X_train_max) - 1 

print(np.min(X_test), np.max(X_test))
from keras.layers import *

from keras.models import Sequential

from keras.optimizers import *

from keras import regularizers

from keras.utils import plot_model, model_to_dot

from IPython.display import SVG
model = Sequential()

l2_reg_conv2d = 0

l2_reg_dense = 0.01

activation_type = 'relu'



model.add(Conv2D(64, kernel_size=3, activation=activation_type, input_shape=(28, 28, 1), padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=3, activation=activation_type, padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=5, activation=activation_type, padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.4))



model.add(Conv2D(128, kernel_size=3, activation=activation_type, padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=3, activation=activation_type, padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=5, activation=activation_type, padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.3))



model.add(Conv2D(256, kernel_size=3, activation=activation_type, padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=3, activation=activation_type, padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=5, activation=activation_type, padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(256, activation=activation_type, kernel_regularizer=regularizers.l2(l2_reg_dense)))

model.add(BatchNormalization())

model.add(Dense(128, activation=activation_type, kernel_regularizer=regularizers.l2(l2_reg_dense)))

model.add(BatchNormalization())

model.add(Dense(10, activation='softmax'))



model.compile(optimizer = SGD(lr=0.01),

              loss = 'sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.summary()
# model_SGD = Sequential()

# l2_reg_conv2d = 0

# l2_reg_dense = 0.01



# model_SGD.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1), padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_SGD.add(BatchNormalization())

# model_SGD.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_SGD.add(BatchNormalization())

# model_SGD.add(Conv2D(64, kernel_size=5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_SGD.add(BatchNormalization())

# model_SGD.add(MaxPooling2D(pool_size=(3, 3)))

# model_SGD.add(Dropout(0.4))



# model_SGD.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_SGD.add(BatchNormalization())

# model_SGD.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_SGD.add(BatchNormalization())

# model_SGD.add(Conv2D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_SGD.add(BatchNormalization())

# model_SGD.add(MaxPooling2D(pool_size=(3, 3)))

# model_SGD.add(Dropout(0.3))



# model_SGD.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_SGD.add(BatchNormalization())

# model_SGD.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_SGD.add(BatchNormalization())

# model_SGD.add(Conv2D(256, kernel_size=5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_SGD.add(BatchNormalization())

# model_SGD.add(MaxPooling2D(pool_size=(3, 3)))

# model_SGD.add(Dropout(0.2))



# model_SGD.add(Flatten())

# model_SGD.add(Dense(256, kernel_regularizer=regularizers.l2(l2_reg_dense)))

# model_SGD.add(BatchNormalization())

# model_SGD.add(Dense(128, kernel_regularizer=regularizers.l2(l2_reg_dense)))

# model_SGD.add(BatchNormalization())

# model_SGD.add(Dense(10, activation='softmax'))



# model_SGD.compile(optimizer = SGD(lr=0.01),

#       loss = 'sparse_categorical_crossentropy',

#       metrics=['accuracy'])



# model_SGD.summary()
# model_SGD_mom = Sequential()

# l2_reg_conv2d = 0

# l2_reg_dense = 0.01



# model_SGD_mom.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1), padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_SGD_mom.add(BatchNormalization())

# model_SGD_mom.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_SGD_mom.add(BatchNormalization())

# model_SGD_mom.add(Conv2D(64, kernel_size=5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_SGD_mom.add(BatchNormalization())

# model_SGD_mom.add(MaxPooling2D(pool_size=(3, 3)))

# model_SGD_mom.add(Dropout(0.4))



# model_SGD_mom.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_SGD_mom.add(BatchNormalization())

# model_SGD_mom.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_SGD_mom.add(BatchNormalization())

# model_SGD_mom.add(Conv2D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_SGD_mom.add(BatchNormalization())

# model_SGD_mom.add(MaxPooling2D(pool_size=(3, 3)))

# model_SGD_mom.add(Dropout(0.3))



# model_SGD_mom.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_SGD_mom.add(BatchNormalization())

# model_SGD_mom.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_SGD_mom.add(BatchNormalization())

# model_SGD_mom.add(Conv2D(256, kernel_size=5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_SGD_mom.add(BatchNormalization())

# model_SGD_mom.add(MaxPooling2D(pool_size=(3, 3)))

# model_SGD_mom.add(Dropout(0.2))



# model_SGD_mom.add(Flatten())

# model_SGD_mom.add(Dense(256, kernel_regularizer=regularizers.l2(l2_reg_dense)))

# model_SGD_mom.add(BatchNormalization())

# model_SGD_mom.add(Dense(128, kernel_regularizer=regularizers.l2(l2_reg_dense)))

# model_SGD_mom.add(BatchNormalization())

# model_SGD_mom.add(Dense(10, activation='softmax'))



# model_SGD_mom.compile(optimizer = SGD(lr=0.01, momentum=0.9),

#       loss = 'sparse_categorical_crossentropy',

#       metrics=['accuracy'])



# model_SGD_mom.summary()
# model_Adadelta = Sequential()

# l2_reg_conv2d = 0

# l2_reg_dense = 0.01



# model_Adadelta.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1), padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_Adadelta.add(BatchNormalization())

# model_Adadelta.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_Adadelta.add(BatchNormalization())

# model_Adadelta.add(Conv2D(64, kernel_size=5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_Adadelta.add(BatchNormalization())

# model_Adadelta.add(MaxPooling2D(pool_size=(3, 3)))

# model_Adadelta.add(Dropout(0.4))



# model_Adadelta.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_Adadelta.add(BatchNormalization())

# model_Adadelta.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_Adadelta.add(BatchNormalization())

# model_Adadelta.add(Conv2D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_Adadelta.add(BatchNormalization())

# model_Adadelta.add(MaxPooling2D(pool_size=(3, 3)))

# model_Adadelta.add(Dropout(0.3))



# model_Adadelta.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_Adadelta.add(BatchNormalization())

# model_Adadelta.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_Adadelta.add(BatchNormalization())

# model_Adadelta.add(Conv2D(256, kernel_size=5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_Adadelta.add(BatchNormalization())

# model_Adadelta.add(MaxPooling2D(pool_size=(3, 3)))

# model_Adadelta.add(Dropout(0.2))



# model_Adadelta.add(Flatten())

# model_Adadelta.add(Dense(256, kernel_regularizer=regularizers.l2(l2_reg_dense)))

# model_Adadelta.add(BatchNormalization())

# model_Adadelta.add(Dense(128, kernel_regularizer=regularizers.l2(l2_reg_dense)))

# model_Adadelta.add(BatchNormalization())

# model_Adadelta.add(Dense(10, activation='softmax'))



# model_Adadelta.compile(optimizer = Adadelta(learning_rate=1.0),

#       loss = 'sparse_categorical_crossentropy',

#       metrics=['accuracy'])



# model_Adadelta.summary()
# model_Adam = Sequential()

# l2_reg_conv2d = 0

# l2_reg_dense = 0.01



# model_Adam.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1), padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_Adam.add(BatchNormalization())

# model_Adam.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_Adam.add(BatchNormalization())

# model_Adam.add(Conv2D(64, kernel_size=5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_Adam.add(BatchNormalization())

# model_Adam.add(MaxPooling2D(pool_size=(3, 3)))

# model_Adam.add(Dropout(0.4))



# model_Adam.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_Adam.add(BatchNormalization())

# model_Adam.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_Adam.add(BatchNormalization())

# model_Adam.add(Conv2D(128, kernel_size=5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_Adam.add(BatchNormalization())

# model_Adam.add(MaxPooling2D(pool_size=(3, 3)))

# model_Adam.add(Dropout(0.3))



# model_Adam.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_Adam.add(BatchNormalization())

# model_Adam.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_Adam.add(BatchNormalization())

# model_Adam.add(Conv2D(256, kernel_size=5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_reg_conv2d)))

# model_Adam.add(BatchNormalization())

# model_Adam.add(MaxPooling2D(pool_size=(3, 3)))

# model_Adam.add(Dropout(0.2))



# model_Adam.add(Flatten())

# model_Adam.add(Dense(256, kernel_regularizer=regularizers.l2(l2_reg_dense)))

# model_Adam.add(BatchNormalization())

# model_Adam.add(Dense(128, kernel_regularizer=regularizers.l2(l2_reg_dense)))

# model_Adam.add(BatchNormalization())

# model_Adam.add(Dense(10, activation='softmax'))



# model_Adam.compile(optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),

#       loss = 'sparse_categorical_crossentropy',

#       metrics=['accuracy'])



# model_Adam.summary()
# from keras.utils import plot_model

# plot_model(model, to_file='model.png')
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping,  ReduceLROnPlateau



datagen = ImageDataGenerator(

    rotation_range = 20,

    width_shift_range = 0.3,

    height_shift_range = 0.3,

    shear_range = 0.2,

    zoom_range = 0.3,

    horizontal_flip = False)
epochs = 75

batch_size = 128



X_train = X_train.reshape(X_train.shape[0],28,28,1)

X_test = X_test.reshape(X_test.shape[0],28,28,1)



train_story = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),

                    epochs = epochs, 

                    steps_per_epoch = 100,

                    validation_data = (X_test, y_test), 

                    callbacks=[

                      ModelCheckpoint('/kaggle/working/best_kannada_model.h5', save_best_only=True),

                      CSVLogger('/kaggle/working/learning_log_RMSprop_without_BN.csv'),

#                       ReduceLROnPlateau(monitor='val_loss', patience=200, verbose=1, factor=0.2),

                      ],

                    verbose=1)
import matplotlib.pyplot as plt 



log_batch_norm = np.array(pd.read_csv("/kaggle/working/learning_log_RMSprop_with_BN.csv")['val_accuracy'])

log_no_batch_norm = np.array(pd.read_csv("/kaggle/working/learning_log_RMSprop_without_BN.csv")['val_accuracy'])



plt.figure(figsize=(20,10))

plt.plot(range(1, 11), log_batch_norm, label='with BatchNorm')

plt.plot(range(1, 11), log_no_batch_norm, label='without BatchNorm')

plt.legend()

plt.show()
import matplotlib.pyplot as plt 



log_softmax = np.array(pd.read_csv("/kaggle/working/learning_log_RMSprop_softmax.csv")['val_accuracy'])

log_elu = np.array(pd.read_csv("/kaggle/working/learning_log_RMSprop_elu.csv")['val_accuracy'])

log_relu = np.array(pd.read_csv("/kaggle/working/learning_log_RMSprop_relu.csv")['val_accuracy'])

log_tanh = np.array(pd.read_csv("/kaggle/working/learning_log_RMSprop_tanh.csv")['val_accuracy'])

log_sigmoid = np.array(pd.read_csv("/kaggle/working/learning_log_RMSprop_sigmoid.csv")['val_accuracy'])

log_exponential = np.array(pd.read_csv("/kaggle/working/learning_log_RMSprop_exponential.csv")['val_accuracy'])



plt.figure(figsize=(20,10))

plt.plot(range(1, 11), log_softmax, label='softmax')

plt.plot(range(1, 11), log_elu, label='elu')

plt.plot(range(1, 11), log_relu, label='relu')

plt.plot(range(1, 11), log_tanh, label='tanh')

plt.plot(range(1, 11), log_sigmoid, label='sigmoid')

plt.plot(range(1, 11), log_exponential, label='exponential')

plt.legend()

plt.show()
# OPTIMIZATORS EXP DO NOT RUN



# epochs = 10

# batch_size = 128



# X_train = X_train.reshape(X_train.shape[0],28,28,1)

# X_test = X_test.reshape(X_test.shape[0],28,28,1)



# train_story_RMSprop = model_RMSprop.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),

#                     epochs = epochs, 

#                     steps_per_epoch = 100,

#                     validation_data = (X_test, y_test), 

#                     callbacks=[

#                       ModelCheckpoint('/kaggle/working/best_kannada_model.h5', save_best_only=True),

#                       CSVLogger('/kaggle/working/learning_log_RMSprop.csv'),

# #                       ReduceLROnPlateau(monitor='val_loss', patience=200, verbose=1, factor=0.2),

#                       ],

#                     verbose=1)



# train_story_SGD_mom = model_SGD_mom.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),

#                     epochs = epochs, 

#                     steps_per_epoch = 100,

#                     validation_data = (X_test, y_test), 

#                     callbacks=[

#                       ModelCheckpoint('/kaggle/working/best_kannada_model.h5', save_best_only=True),

#                       CSVLogger('/kaggle/working/learning_log_SGD_mom.csv'),

#                       ],

#                     verbose=1)



# train_story_SGD = model_SGD.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),

#                     epochs = epochs, 

#                     steps_per_epoch = 100,

#                     validation_data = (X_test, y_test), 

#                     callbacks=[

#                       ModelCheckpoint('/kaggle/working/best_kannada_model.h5', save_best_only=True),

#                       CSVLogger('/kaggle/working/learning_log_SGD.csv'),

#                       ],

#                     verbose=1)



# train_story_Adam = model_Adam.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),

#                     epochs = epochs, 

#                     steps_per_epoch = 100,

#                     validation_data = (X_test, y_test), 

#                     callbacks=[

#                       ModelCheckpoint('/kaggle/working/best_kannada_model.h5', save_best_only=True),

#                       CSVLogger('/kaggle/working/learning_log_Adam.csv'),

#                       ],

#                     verbose=1)



# train_story_Adadelta = model_Adadelta.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),

#                     epochs = epochs, 

#                     steps_per_epoch = 100,

#                     validation_data = (X_test, y_test), 

#                     callbacks=[

#                       ModelCheckpoint('/kaggle/working/best_kannada_model.h5', save_best_only=True),

#                       CSVLogger('/kaggle/working/learning_log_Adadelta.csv'),

#                       ],

#                     verbose=1)
log_SGD = np.array(pd.read_csv("/kaggle/working/learning_log_SGD.csv")['val_accuracy'])

log_SGD_mom = np.array(pd.read_csv("/kaggle/working/learning_log_SGD_mom.csv")['val_accuracy'])

log_Adam = np.array(pd.read_csv("/kaggle/working/learning_log_Adam.csv")['val_accuracy'])

log_Adadelta = np.array(pd.read_csv("/kaggle/working/learning_log_Adadelta.csv")['val_accuracy'])

log_RMSprop = np.array(pd.read_csv("/kaggle/working/learning_log_RMSprop.csv")['val_accuracy'])
import matplotlib.pyplot as plt 



plt.figure(figsize=(20,10))

plt.plot(range(1, 11), log_SGD, label='SGD')

plt.plot(range(1, 11), log_SGD_mom, label='SGD_mom')

plt.plot(range(1, 11), log_Adam, label='Adam')

plt.plot(range(1, 11), log_Adadelta, label='Adadelta')

plt.plot(range(1, 11), log_RMSprop, label='RMSprop')

plt.legend()

plt.show()
test_csv = pd.read_csv("../input/Kannada-MNIST/test.csv")

X_val = np.array(test_csv.drop("id",axis=1), dtype=np.float32)

X_val.shape
X_val_max = np.max(X_val)

X_val = X_val / (0.5 * X_val_max) - 1

X_val = np.reshape(X_val, (-1,28,28,1))



print(X_val.shape, np.min(X_val), np.max(X_val))
from keras.models import load_model



best_model = load_model('/kaggle/working/best_kannada_model.h5')

Y_val = best_model.predict(X_val)

Y_val = np.argmax(Y_val, axis = 1)
submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

submission['label'] = Y_val

submission.to_csv("submission.csv",index=False)