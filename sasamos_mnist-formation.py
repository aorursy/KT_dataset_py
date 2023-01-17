import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split

import keras

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization

from keras.models import Sequential

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from keras.optimizers import Adam, RMSprop

from keras.preprocessing.image import ImageDataGenerator

tf.set_random_seed(42)
# Settings

train_path = os.path.join('..', 'input', 'train.csv')

test_path = os.path.join('..', 'input', 'test.csv')



# CNN model settings

size = 28

lr = 0.001

num_classes = 10



# Training settings

epochs = 30

batch_size = 128
# data loading

raw_train_df = pd.read_csv(train_path)

raw_test_df = pd.read_csv(test_path)
def parse_train_df(_train_df):

    labels = _train_df.iloc[:,0].values

    imgs = _train_df.iloc[:,1:].values

    imgs_2d = np.array([[[[float(imgs[index][i*28 + j]) / 255] for j in range(28)] for i in range(28)] for index in range(len(imgs))])

    processed_labels = [[0 for _ in range(10)] for i in range(len(labels))]

    for i in range(len(labels)):

        processed_labels[i][labels[i]] = 1

    return np.array(processed_labels), imgs_2d



def parse_test_df(test_df):

    imgs = test_df.iloc[:, 0:].values

    imgs_2d = np.array([[[[float(imgs[index][i * 28 + j]) / 255] for j in range(28)] for i in range(28)] for index in

                        range(len(imgs))])

    return imgs_2d
# Data preprocessing

y_train_set, x_train_set = parse_train_df(raw_train_df)

x_test = parse_test_df(raw_test_df)



x_train, x_val, y_train, y_val = train_test_split(x_train_set, y_train_set, test_size=0.20, random_state=42)
# Training data insights

raw_train_df['label'].value_counts().plot.bar()
print("Number of 1: {}".format(len(raw_train_df[raw_train_df['label'] == 1])))

print("Number of 5: {}".format(len(raw_train_df[raw_train_df['label'] == 5])))
# Image visualization

n = 5

fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))

for i in range(n**2):

    ax = axs[i // n, i % n]

    (-x_train[i]+1)/2

    ax.imshow((-x_train[i, :, :, 0] + 1)/2, cmap=plt.cm.gray)

    ax.axis('off')

plt.tight_layout()

plt.show()
# CNN model

model = keras.Sequential()



model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),

                 activation='relu',

                 input_shape=(size, size, 1)))

model.add(Conv2D(32, (3, 3), activation='relu', strides=(2, 2)))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu', strides=(2, 2)))

model.add(BatchNormalization())

model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(BatchNormalization())



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(128, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=Adam(lr),

              metrics=['accuracy'])



checkpoint = ModelCheckpoint('model_ckpt.{epoch:02d}.hdf5',

                                             save_best_only=True,

                                             save_weights_only=True)

lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3,

                      mode='max', cooldown=3, verbose=1)

callback_list = [checkpoint, lr_reducer]
# Training

training_history = model.fit(

    x_train,

    y_train,

    epochs=epochs,

    verbose=1,

    validation_data=(x_val, y_val),

    callbacks=callback_list

)
# Training recap

epoch_range = [e for e in range(1, epochs + 1)]

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(24, 4))



axs[0].plot(epoch_range,

         training_history.history['loss'],

         training_history.history['val_loss'],

)

axs[0].set_title('Training loss')

axs[1].plot(epoch_range,

         training_history.history['acc'],

         training_history.history['val_acc'],

)

axs[1].set_title('Training accuracy')



plt.show()
# Creating image generator

image_generator = ImageDataGenerator(

    rotation_range=15,

    width_shift_range=0.1,

    height_shift_range=0.1,

    zoom_range=0.1

)



image_generator.fit(x_train)
# Retraining the same model

model_augmented = keras.Sequential()



model_augmented.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),

                 activation='relu',

                 input_shape=(size, size, 1)))

model_augmented.add(Conv2D(32, (3, 3), activation='relu', strides=(2, 2)))

model_augmented.add(BatchNormalization())

model_augmented.add(Dropout(0.3))



model_augmented.add(Conv2D(64, (3, 3), activation='relu'))

model_augmented.add(Conv2D(64, (3, 3), activation='relu', strides=(2, 2)))

model_augmented.add(BatchNormalization())

model_augmented.add(Dropout(0.3))

model_augmented.add(Conv2D(128, (3, 3), activation='relu'))

model_augmented.add(BatchNormalization())



model_augmented.add(Flatten())

model_augmented.add(Dense(256, activation='relu'))

model_augmented.add(Dropout(0.25))

model_augmented.add(Dense(128, activation='relu'))

model_augmented.add(Dense(num_classes, activation='softmax'))



model_augmented.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=RMSprop(0.001),

              metrics=['accuracy'])



aug_result = model_augmented.fit_generator(

    image_generator.flow(x_train, y_train, batch_size=batch_size),

    epochs=epochs,

    steps_per_epoch=len(x_train) // batch_size,

    verbose=1,

    validation_data=(x_val, y_val),

    callbacks=callback_list

)



model_augmented.save('mnist_model.h5')
# Training recap of your augmented dataset

epoch_range = [e for e in range(1, epochs + 1)]

plt.plot(epoch_range,

         aug_result.history['acc'],

         aug_result.history['val_acc'],

)

plt.title('Accuracy')

plt.show()
# Prediction

pred = model.predict(x_test)

pred_aug = model_augmented.predict(x_test)
test_pred = pd.DataFrame(model.predict_classes(x_test), columns=['Label'])

test_pred.index.name = 'ImageId'

test_pred.to_csv('mnist_submission.csv')