# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import utils as tf_utils

from keras.callbacks.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense, Lambda

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization

from keras import regularizers





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
mnist_train_complete = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

mnist_test_complete = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")



mnist_train_complete.head(5)
# preparing the training and testing sets, separating the training pictures of the numbers (i.e. train_x)

# from their label (i.e train_y).

# We set here also the data types as int32

train_y = mnist_train_complete.iloc[:, 0].values.astype('int32')

train_x = mnist_train_complete.iloc[:, 1:].values.astype('float32')

test_x = mnist_test_complete.values.astype('float32')



# reshaping the training and testing sets to have each digit image of 28 by 28 pixels

train_x = train_x.reshape(train_x.shape[0], 28, 28)

test_x = test_x.reshape(test_x.shape[0], 28, 28)
for i in range (10,14):

    plt.subplot(330 + i+1)

    plt.imshow(train_x[i], cmap=plt.get_cmap('gray'))

    plt.title(train_y[i])
def visualize_detail(img, ax):

    ax.imshow(img, cmap='gray')

    width, height = img.shape

    threshold = img.max()/2.5

    for x in range(width):

        for y in range(height):

            ax.annotate(str(round(img[x][y], 2)), xy=(y,x),

                        horizontalalignment='center',

                        verticalalignment='center',

                        color='white' if img[x][y] < threshold else 'black')



fig = plt.figure(figsize=(12,12))

ax = fig.add_subplot(111)

    

visualize_detail(train_x[10], ax)
# Normalizing the training and testing sets

train_x = train_x.astype('float32')/np.max(train_x)

test_x = test_x.astype('float32')/np.max(test_x)



# center the normalized data around zero

mean = np.std(train_x)

train_x -= mean

mean = np.std(test_x)

test_x -= mean
# creating the training and validationg sets

splitted_train_X, splitted_test_X, splitted_train_y, splitted_test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=81)



# one-hot encoding the training and validation sets

ohe_splitted_train_y = tf_utils.to_categorical(splitted_train_y, 10)

ohe_splitted_test_y = tf_utils.to_categorical(splitted_test_y, 10)



# print first one-hot training labels

print('One-hot labels:')

print(splitted_train_y[:10])
# define a fully connected NNs model

model_sol_1 = tf.keras.models.Sequential()

model_sol_1.add(tf.keras.layers.Flatten(input_shape = splitted_train_X.shape[1:]))

model_sol_1.add(tf.keras.layers.Dense(512, activation='relu'))

model_sol_1.add(tf.keras.layers.Dropout(0.2))

model_sol_1.add(tf.keras.layers.Dense(512, activation='relu'))

model_sol_1.add(tf.keras.layers.Dropout(0.2))

model_sol_1.add(tf.keras.layers.Dense(10, activation='softmax'))



# summary of model

model_sol_1.summary()
# compile the model

model_sol_1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# evaluate test accuracy

score = model_sol_1.evaluate(splitted_test_X, ohe_splitted_test_y, verbose=0)

accuracy = 100 * score[1]



# print test accuracy

print('Test accuracy: %4f%%' % accuracy)
# checkpointer to save the best weihts

checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)



hist_sol_1 = model_sol_1.fit(splitted_train_X, ohe_splitted_train_y, batch_size=128, epochs=10,

                 validation_split=0.2, callbacks=[checkpointer],

                 verbose=2, shuffle=True)
# plot the losses

plt.figure(figsize=(10,5))

plt.plot(hist_sol_1.history['loss'], linestyle="--")

plt.plot(hist_sol_1.history['val_loss'], linestyle="-.")

plt.title('model losses')

plt.ylabel('loss')

plt.xlabel('epochs')

plt.legend(["loss", "val_loss"], loc='upper left')

axes = plt.gca()

plt.show()
#load the weights that resulted in the minimal validation loss

model_sol_1.load_weights('mnist.model.best.hdf5')



score = model_sol_1.evaluate(splitted_test_X, ohe_splitted_test_y, verbose=0)

accuracy = 100 * score[1]



#print test accuracy

print('Test accuracy: %.4f%%' % accuracy)
predictions = model_sol_1.predict(test_x)

predictions = [ np.argmax(x) for x in predictions ]
# prepare submission

submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submission.drop('Label', axis=1, inplace=True)

submission['Label'] = predictions

submission.to_csv('submission1.csv', index=False)
extended_splitted_train_X = splitted_train_X[..., tf.newaxis]

extended_splitted_test_X = splitted_test_X[..., tf.newaxis]

extended_splitted_test_X.shape
# define a Convolutional NNs model

model_sol_2 = Sequential()

model_sol_2.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=extended_splitted_train_X.shape[1:]))

model_sol_2.add(MaxPooling2D(pool_size=2))

model_sol_2.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))

model_sol_2.add(MaxPooling2D(pool_size=2))

model_sol_2.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))

model_sol_2.add(MaxPooling2D(pool_size=2))



# Converts our 3D feature maps to 1D features vectors

model_sol_2.add(Flatten())

model_sol_2.add(Dense(64))

model_sol_2.add(Activation('relu'))

model_sol_2.add(Dropout(0.2))

model_sol_2.add(Dense(10, activation='softmax'))



# summary of model

#model_sol_2.summary()
# compile the model

model_sol_2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# evaluate test accuracy

score = model_sol_2.evaluate(extended_splitted_test_X, ohe_splitted_test_y, verbose=0)

accuracy = 100 * score[1]



# print test accuracy

print('Test accuracy: %4f%%' % accuracy)
# checkpointer to save the best weihts

checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)



hist_sol_2 = model_sol_2.fit(extended_splitted_train_X, ohe_splitted_train_y, batch_size=128,

                             epochs=10, callbacks=[checkpointer],

                             verbose=2, validation_data=(extended_splitted_test_X, ohe_splitted_test_y), shuffle=True)
# plot the losses

plt.figure(figsize=(10,5))

plt.plot(hist_sol_2.history['loss'], linestyle="--")

plt.plot(hist_sol_2.history['val_loss'], linestyle="-.")

plt.title('model losses')

plt.ylabel('loss')

plt.xlabel('epochs')

plt.legend(["loss", "val_loss"], loc='upper left')

axes = plt.gca()

plt.show()
#load the weights that resulted in the minimal validation loss

model_sol_2.load_weights('mnist.model.best.hdf5')



score = model_sol_2.evaluate(extended_splitted_test_X, ohe_splitted_test_y, verbose=0)

accuracy = 100 * score[1]



#print test accuracy

print('Test accuracy: %.4f%%' % accuracy)
# extend the test imagae set with an additional dimension

extended_test_x = test_x[..., tf.newaxis]

predictions = model_sol_2.predict(extended_test_x)

predictions = [ np.argmax(x) for x in predictions ]



# prepare submission

submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submission.drop('Label', axis=1, inplace=True)

submission['Label'] = predictions

submission.to_csv('submission2.csv', index=False)
# define a data augmentator for our images

image_augmentator = ImageDataGenerator(

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    # rescale=1./255,

    shear_range=0.2,

    zoom_range=0.1,

    fill_mode='nearest')



# define size of batch

batch_size = 32



train_batches = image_augmentator.flow(extended_splitted_train_X, ohe_splitted_train_y, batch_size=batch_size)

val_batches = image_augmentator.flow(extended_splitted_test_X, ohe_splitted_test_y, batch_size=batch_size)
example_img = train_x[10][..., tf.newaxis]

transf_params = { 'theta':15., 'tx':0.1, 'ty':0.1, 'shear':0.2 }

augmented_image = image_augmentator.apply_transform(example_img, transf_params)



# reducing dimensinoality to two

twoDim_image = augmented_image[:, :, 0]



fig = plt.figure(figsize=(12,12))

ax = fig.add_subplot(111)

visualize_detail(twoDim_image, ax)
# define a Convolutional NNs model (solution number 3)

model_sol_3 = Sequential()

model_sol_3.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=extended_splitted_train_X.shape[1:]))

model_sol_3.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))

model_sol_3.add(MaxPooling2D(pool_size=2))

model_sol_3.add(Dropout(0.1))

model_sol_3.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))

model_sol_3.add(MaxPooling2D(pool_size=2))



# Converts our 3D feature maps to 1D features vectors

model_sol_3.add(Flatten())

model_sol_3.add(Dense(64))

model_sol_3.add(Activation('relu'))

model_sol_3.add(Dropout(0.2))

model_sol_3.add(Dense(10, activation='softmax'))



# summary of model

#model_sol_3.summary()
# compile the model

model_sol_3.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# checkpointer to save the best weihts

checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)



hist_sol_3 = model_sol_3.fit_generator(generator=train_batches, steps_per_epoch =extended_splitted_train_X.shape[0] // batch_size,

                                       epochs=32, callbacks=[checkpointer],

                                       validation_data=val_batches, validation_steps=extended_splitted_test_X.shape[0] // batch_size,

                                       verbose=2)
# plot the losses

plt.figure(figsize=(10,5))

plt.plot(hist_sol_3.history['loss'], linestyle="--")

plt.plot(hist_sol_3.history['val_loss'], linestyle="-.")

plt.title('model losses')

plt.ylabel('loss')

plt.xlabel('epochs')

plt.legend(["loss", "val_loss"], loc='upper left')

axes = plt.gca()

plt.show()
#load the weights that resulted in the minimal validation loss

model_sol_3.load_weights('mnist.model.best.hdf5')



score = model_sol_3.evaluate(extended_splitted_test_X, ohe_splitted_test_y, verbose=0)

accuracy = 100 * score[1]



#print test accuracy

print('Test accuracy: %.4f%%' % accuracy)
# extend the test imagae set with an additional dimension

extended_test_x = test_x[..., tf.newaxis]

predictions = model_sol_3.predict(extended_test_x)

predictions = [ np.argmax(x) for x in predictions ]



# prepare submission

submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submission.drop('Label', axis=1, inplace=True)

submission['Label'] = predictions

submission.to_csv('submission3.csv', index=False)
# define a Convolutional NNs model (solution number 4)

model_sol_4_1 = Sequential()

model_sol_4_1.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=extended_splitted_train_X.shape[1:]))

model_sol_4_1.add(BatchNormalization())

model_sol_4_1.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))

model_sol_4_1.add(MaxPooling2D(pool_size=2))

model_sol_4_1.add(Dropout(0.1))

model_sol_4_1.add(BatchNormalization())

model_sol_4_1.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))

model_sol_4_1.add(MaxPooling2D(pool_size=2))



# Converts our 3D feature maps to 1D features vectors

model_sol_4_1.add(Flatten())

model_sol_4_1.add(BatchNormalization())

model_sol_4_1.add(Dense(64))

model_sol_4_1.add(Activation('relu'))

model_sol_4_1.add(Dropout(0.2))

model_sol_4_1.add(BatchNormalization())

model_sol_4_1.add(Dense(10, activation='softmax'))



# summary of model

#model_sol_4_1.summary()
# compile the model

model_sol_4_1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# checkpointer to save the best weihts

checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)



hist_sol_4 = model_sol_4_1.fit_generator(generator=train_batches, steps_per_epoch =extended_splitted_train_X.shape[0] // batch_size,

                                       epochs=32, callbacks=[checkpointer],

                                       validation_data=val_batches, validation_steps=extended_splitted_test_X.shape[0] // batch_size,

                                       verbose=2)
# plot the losses

plt.figure(figsize=(10,5))

plt.plot(hist_sol_4.history['loss'], linestyle="--")

plt.plot(hist_sol_4.history['val_loss'], linestyle="-.")

plt.title('model losses')

plt.ylabel('loss')

plt.xlabel('epochs')

plt.legend(["loss", "val_loss"], loc='upper left')

axes = plt.gca()

plt.show()
#load the weights that resulted in the minimal validation loss

model_sol_4_1.load_weights('mnist.model.best.hdf5')



score = model_sol_4_1.evaluate(extended_splitted_test_X, ohe_splitted_test_y, verbose=0)

accuracy = 100 * score[1]



#print test accuracy

print('Test accuracy: %.4f%%' % accuracy)
model_sol_4_2 = Sequential()

model_sol_4_2.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=extended_splitted_train_X.shape[1:]))

model_sol_4_2.add(BatchNormalization())

model_sol_4_2.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))

model_sol_4_2.add(MaxPooling2D(pool_size=2))

model_sol_4_2.add(Dropout(0.1))

model_sol_4_2.add(BatchNormalization())

model_sol_4_2.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))

model_sol_4_2.add(MaxPooling2D(pool_size=2))



# Converts our 3D feature maps to 1D features vectors

model_sol_4_2.add(Flatten())

model_sol_4_2.add(BatchNormalization())

model_sol_4_2.add(Dense(64))

model_sol_4_2.add(Activation('relu'))

model_sol_4_2.add(Dropout(0.2))

model_sol_4_2.add(BatchNormalization())

model_sol_4_2.add(Dense(10, activation='softmax'))



# summary of model

#model_sol_4_2.summary()
# compile the model

model_sol_4_2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# checkpointer to save the best weihts

checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)



hist_sol_4 = model_sol_4_2.fit_generator(generator=train_batches, steps_per_epoch=extended_splitted_train_X.shape[0] // batch_size,

                                       epochs=32, callbacks=[checkpointer],

                                       validation_data=val_batches, validation_steps=extended_splitted_test_X.shape[0] // batch_size,

                                       verbose=2)
# plot the losses

plt.figure(figsize=(10,5))

plt.plot(hist_sol_4.history['loss'], linestyle="--")

plt.plot(hist_sol_4.history['val_loss'], linestyle="-.")

plt.title('model losses')

plt.ylabel('loss')

plt.xlabel('epochs')

plt.legend(["loss", "val_loss"], loc='upper left')

axes = plt.gca()

plt.show()
#load the weights that resulted in the minimal validation loss

model_sol_4_2.load_weights('mnist.model.best.hdf5')



score = model_sol_4_2.evaluate(extended_splitted_test_X, ohe_splitted_test_y, verbose=0)

accuracy = 100 * score[1]



#print test accuracy

print('Test accuracy: %.4f%%' % accuracy)
extended_test_x = test_x[..., tf.newaxis]

predictions = model_sol_4_2.predict(extended_test_x)

predictions = [ np.argmax(x) for x in predictions ]



# prepare submission

submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submission.drop('Label', axis=1, inplace=True)

submission['Label'] = predictions

submission.to_csv('submission4.csv', index=False)
# create new datasets out of the original files provided by kaggle (to avoid confussions with other

# variables created in other sections of this notebook and because I need this data without the first preprocessing steps

# I performed in my previous solutions)

train_y_sol5 = mnist_train_complete.iloc[:, 0].values.astype('int32')

train_x_sol5 = mnist_train_complete.iloc[:, 1:].values.astype('float32')

test_x_sol5 =  mnist_test_complete.values.astype('float32')



# reshaping the new training and testing sets to have each digit image of 28 by 28 pixels

train_x_sol5 = train_x_sol5.reshape(train_x_sol5.shape[0], 28, 28)

test_x_sol5 = test_x_sol5.reshape(test_x_sol5.shape[0], 28, 28)



# add another dimension to the training data

train_x_sol5 = train_x_sol5[..., tf.newaxis]

test_x_sol5  = test_x_sol5[..., tf.newaxis]
# new preprocessing of data (to be applied to each individual image by the Lamda layer)

mean_px = train_x_sol5.mean().astype(np.float32)

std_px = train_x_sol5.std().astype(np.float32)



# define the function that will be performed by our Lambda layer on each of the input images

def standardize(x): 

    return (x-mean_px)/std_px
# cross validation

s5_train_x, s5_test_x, s5_train_y, s5_test_y = train_test_split(train_x_sol5, train_y_sol5,

                                                                test_size=0.2,

                                                                random_state=81)

# one-hot encoding the target labels

ohe_s5_train_y = tf_utils.to_categorical(s5_train_y, 10)

ohe_s5_test_y = tf_utils.to_categorical(s5_test_y, 10)



# create new image generators using the same image_augmentator created previously,

# but with a different number of batches (prevous batch size was 32).

train_batches_sol5 = image_augmentator.flow(s5_train_x, ohe_s5_train_y, batch_size=64)

val_batches_sol5 = image_augmentator.flow(s5_test_x, ohe_s5_test_y, batch_size=64)
model_sol_5 = Sequential()

model_sol_5.add(Lambda(standardize, input_shape=(28,28,1)))

model_sol_5.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',

                 kernel_regularizer=regularizers.l2(0.1),

                 ))

model_sol_5.add(BatchNormalization())

model_sol_5.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'

                ))

model_sol_5.add(MaxPooling2D(pool_size=2))

#model.add(Dropout(0.1))



model_sol_5.add(BatchNormalization())

model_sol_5.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'

         ))

model_sol_5.add(MaxPooling2D(pool_size=2))



# Converts our 3D feature maps to 1D features vectors

model_sol_5.add(Flatten())

model_sol_5.add(BatchNormalization())

model_sol_5.add(Dense(64))

model_sol_5.add(Activation('relu'))

model_sol_5.add(Dropout(0.2))

model_sol_5.add(BatchNormalization())

model_sol_5.add(Dense(10, activation='softmax'))



# summary of model

#model_sol_5.summary()
# compile the model

model_sol_5.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
# checkpointer to save the best weihts

checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)



hist_sol_5 = model_sol_5.fit_generator(generator=train_batches_sol5, steps_per_epoch=s5_train_x.shape[0] // 64,

                                       epochs=32, callbacks=[checkpointer],

                                       validation_data=val_batches_sol5, validation_steps=s5_test_x.shape[0] // 64, verbose=2)
# plot the losses

plt.figure(figsize=(10,5))

plt.plot(hist_sol_5.history['loss'], linestyle="--")

plt.plot(hist_sol_5.history['val_loss'], linestyle="-.")

plt.title('model losses')

plt.ylabel('loss')

plt.xlabel('epochs')

plt.legend(["loss", "val_loss"], loc='upper left')

axes = plt.gca()

plt.show()
#load the weights that resulted in the minimal validation loss

model_sol_5.load_weights('mnist.model.best.hdf5')



score = model_sol_5.evaluate(s5_test_x, ohe_s5_test_y, verbose=0)

accuracy = 100 * score[1]



#print test accuracy

print('Test accuracy: %.4f%%' % accuracy)
predictions = model_sol_5.predict(test_x_sol5)

predictions = [ np.argmax(x) for x in predictions ]



# prepare submission

submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submission.drop('Label', axis=1, inplace=True)

submission['Label'] = predictions

submission.to_csv('submission5.csv', index=False)
model_sol_6 = Sequential()

model_sol_6.add(Lambda(standardize, input_shape=(28,28,1)))

model_sol_6.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))

model_sol_6.add(BatchNormalization())

model_sol_6.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))

model_sol_6.add(MaxPooling2D(pool_size=2))

model_sol_6.add(Dropout(0.1))

model_sol_6.add(BatchNormalization())

model_sol_6.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))

model_sol_6.add(MaxPooling2D(pool_size=2))



# Converts our 3D feature maps to 1D features vectors

model_sol_6.add(Flatten())

model_sol_6.add(BatchNormalization())

model_sol_6.add(Dense(64))

model_sol_6.add(Activation('relu'))

model_sol_6.add(Dropout(0.2))

model_sol_6.add(BatchNormalization())

model_sol_6.add(Dense(10, activation='softmax'))



# summary of model

#model_sol_6.summary()
# compile the model

model_sol_6.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
# checkpointer to save the best weihts

checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)



hist_sol_6 = model_sol_6.fit_generator(generator=train_batches_sol5, steps_per_epoch=s5_train_x.shape[0] // 64,

                                       epochs=32, callbacks=[checkpointer],

                                       validation_data=val_batches_sol5, validation_steps=s5_test_x.shape[0] // 64, verbose=2)
# plot the losses

plt.figure(figsize=(10,5))

plt.plot(hist_sol_6.history['loss'], linestyle="--")

plt.plot(hist_sol_6.history['val_loss'], linestyle="-.")

plt.title('model losses')

plt.ylabel('loss')

plt.xlabel('epochs')

plt.legend(["loss", "val_loss"], loc='upper left')

axes = plt.gca()

plt.show()
#load the weights that resulted in the minimal validation loss

model_sol_6.load_weights('mnist.model.best.hdf5')



score = model_sol_6.evaluate(s5_test_x, ohe_s5_test_y, verbose=0)

accuracy = 100 * score[1]



#print test accuracy

print('Test accuracy: %.4f%%' % accuracy)
model_sol_6.optimizer.lerning_rate=0.01

gen = ImageDataGenerator()

batches = gen.flow(train_x_sol5, tf_utils.to_categorical(train_y_sol5, 10), batch_size=64)

hist_sol_6 = model_sol_6.fit_generator(generator=batches, steps_per_epoch=train_x_sol5.shape[0] // 64,

                          epochs=50, verbose=2)

# I didn't use a callback on this training step becuase the 'checkpointer' callback I defined works only when

# the model produces validation loss metrics. In order to do that, I need to pass validation data to the

# fit_generator method. For this second training step I did not pass such validation data becase we do not have 

# test data to validate against - now I am using the complete set of images provided by kaggle.
predictions = model_sol_6.predict(test_x_sol5)

predictions = [ np.argmax(x) for x in predictions ]



# prepare submission

submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submission.drop('Label', axis=1, inplace=True)

submission['Label'] = predictions

submission.to_csv('submission6.csv', index=False)
os.remove('submission1.csv')

os.remove('submission2.csv')

os.remove('submission3.csv')

os.remove('submission4.csv')

os.remove('submission5.csv')

os.remove('submission6.csv')
final_train_x = train_x[..., tf.newaxis]

final_ohe_train_y = tf_utils.to_categorical(train_y, 10)

final_train_batches = image_augmentator.flow(final_train_x, final_ohe_train_y, batch_size=64)



final_model = Sequential()

final_model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=extended_splitted_train_X.shape[1:]))

final_model.add(BatchNormalization())

final_model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))

final_model.add(MaxPooling2D(pool_size=2))

final_model.add(Dropout(0.1))

final_model.add(BatchNormalization())

final_model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))

final_model.add(MaxPooling2D(pool_size=2))



# Converts our 3D feature maps to 1D features vectors

final_model.add(Flatten())

final_model.add(BatchNormalization())

final_model.add(Dense(64))

final_model.add(Activation('relu'))

final_model.add(Dropout(0.2))

final_model.add(BatchNormalization())

final_model.add(Dense(10, activation='softmax'))



# compile the model

final_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])



final_model.fit_generator(generator=final_train_batches, steps_per_epoch=final_train_batches.n,

                          epochs=1, verbose=1)
predictions = final_model.predict(test_x[..., tf.newaxis])

predictions = [ np.argmax(x) for x in predictions ]



# prepare submission

submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submission.drop('Label', axis=1, inplace=True)

submission['Label'] = predictions

submission.to_csv('submission.csv', index=False)