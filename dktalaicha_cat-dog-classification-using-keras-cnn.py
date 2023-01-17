# Import necessary libraries

from matplotlib import pyplot

import pandas as pd

import numpy as np

import seaborn as sns

import warnings

import os

import pickle

import random

from shutil import copyfile

from random import random, seed

from matplotlib.image import imread



warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:,.2f}'.format

pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 200)





# baseline model with dropout for the dogs vs cats dataset

import sys

from matplotlib import pyplot

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Dropout

from keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ReduceLROnPlateau



# make a prediction for a new image.

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.models import load_model
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Unzip the train and test data sets

!unzip '../input/dogs-vs-cats/train.zip' -d '/tmp/cats-vs-dogs'

!unzip '../input/dogs-vs-cats/test1.zip' -d '/tmp/cats-vs-dogs'
base_dir = '/tmp/cats-vs-dogs/'



train_dir = '/tmp/cats-vs-dogs/train/'

test_dir = '/tmp/cats-vs-dogs/test1/'
# Creating a dataframe for train set

filenames = os.listdir(train_dir)

categories = []

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})
df.head()
pyplot.figure(figsize=(8,6))

sns.countplot(x='category',data=df)

pyplot.show()
df['category'].value_counts()
# plot dog photos from the dogs vs cats dataset

fig=pyplot.figure(figsize=(16, 16))

columns = 4

rows = 5

i=1

for filename in os.listdir(train_dir)[0:20]:

    image = imread(train_dir + filename)

    fig.add_subplot(rows, columns, i)

    pyplot.imshow(image)

    i+=1

pyplot.show()
# plot first few dog images

pyplot.figure(figsize=(12,12))

for i in range(9):

    # define subplot

    pyplot.subplot(330 + 1 + i)

    # define filename

    filename = train_dir + 'dog.' + str(i) + '.jpg'

    # load image pixels

    image = imread(filename)

    # plot raw pixel data

    pyplot.imshow(image)

# show the figure

pyplot.show()
# plot first few cat images

pyplot.figure(figsize=(12,12))

for i in range(9):

    # define subplot

    pyplot.subplot(330 + 1 + i)

    # define filename

    filename = train_dir + 'cat.' + str(i) + '.jpg'

    # load image pixels

    image = imread(filename)

    # plot raw pixel data

    pyplot.imshow(image)

# show the figure

pyplot.show()
# create directories

subdirs = ['training/', 'validation/']

for subdir in subdirs:

    # create label subdirectories

    labeldirs = ['dogs/', 'cats/']

    for labldir in labeldirs:

        newdir = base_dir + subdir + labldir

        os.makedirs(newdir, exist_ok=True)
os.listdir(base_dir)
# seed random number generator

seed(1)

# define ratio of pictures to use for validation

val_ratio = 0.25

# copy training dataset images into subdirectories

src_directory = 'train/'

for file in os.listdir(base_dir + src_directory):

    src = base_dir + src_directory + '/' + file

    dst_dir = base_dir + 'training/'

    if random() < val_ratio:

        dst_dir = base_dir + 'validation/'

    if file.startswith('cat'):

        dst =  dst_dir + 'cats/'  + file

        copyfile(src, dst)

    elif file.startswith('dog'):

        dst =  dst_dir + 'dogs/'  + file

        copyfile(src, dst)
# Now, Let's look at the filenames in cats and dogs training and validation directories.

print(os.listdir('/tmp/cats-vs-dogs/training/cats')[0:5])

print(os.listdir('/tmp/cats-vs-dogs/training/dogs')[0:5])

print(os.listdir('/tmp/cats-vs-dogs/validation/cats')[0:5])

print(os.listdir('/tmp/cats-vs-dogs/validation/dogs')[0:5])
train_dir = '/tmp/cats-vs-dogs/training/'

valid_dir = '/tmp/cats-vs-dogs/validation/'
# define cnn model



model = Sequential()

    

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', name='conv_1', padding='same', input_shape=(200, 200, 3)))

model.add(MaxPooling2D((2, 2), name='maxpool_1'))

model.add(Dropout(0.2))

    

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', name='conv_2', padding='same'))

model.add(MaxPooling2D((2, 2), name='maxpool_2'))

model.add(Dropout(0.2))

    

model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', name='conv_3', padding='same'))

model.add(MaxPooling2D((2, 2), name='maxpool_3'))

model.add(Dropout(0.2))



model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', name='conv_4', padding='same'))

model.add(MaxPooling2D((2, 2), name='maxpool_4'))

model.add(Dropout(0.2))

    

model.add(Flatten())

model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', name='dense_1'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid', name='output'))

    

# compile model

opt = SGD(lr=0.001, momentum=0.9)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
# Summary of the Model

print(model.summary())
# Data Augmentation

# datagen = ImageDataGenerator(rescale=1.0/255.0)

    

# # prepare iterator

# train_it = datagen.flow_from_directory(train_dir,class_mode='binary', batch_size=64, target_size=(200, 200))

# valid_it = datagen.flow_from_directory(valid_dir,class_mode='binary', batch_size=64, target_size=(200, 200))



# create data generators

train_datagen = ImageDataGenerator(rescale=1.0/255.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1.0/255.0)



# prepare iterators

train_it = train_datagen.flow_from_directory(train_dir, class_mode='binary', batch_size=64, target_size=(200, 200))

valid_it = valid_datagen.flow_from_directory(valid_dir, class_mode='binary', batch_size=64, target_size=(200, 200))
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.001)
callbacks = [earlystop, learning_rate_reduction]
# fit model

history = model.fit_generator(train_it, steps_per_epoch=len(train_it),validation_data=valid_it, 

                              validation_steps=len(valid_it), epochs=70, callbacks=callbacks, verbose=1) 
# save model

model.save('final_model.h5')   
# evaluate model

_, acc = model.evaluate_generator(valid_it, steps=len(valid_it), verbose=1)

print('> %.3f' % (acc * 100.0))
# A figure is also created showing a line plot for the loss and another for the accuracy of the model 

# on both the training (red) and validation (blue) datasets.



# plot diagnostic learning curves

acc = history.history["accuracy"]

val_acc = history.history["val_accuracy"]



loss = history.history["loss"]

val_loss = history.history["val_loss"]

epochs = range(len(acc))



fig, (ax1, ax2) = pyplot.subplots(2, 1, figsize=(12, 12))



# plot accuracy

ax1.plot(epochs, acc, "r", label="Training accuracy")

ax1.plot(epochs, val_acc, "b", label="Validation accuracy")

ax1.title.set_text("Training and Validation accuracy")



# plot loss

ax2.plot(epochs, loss, "r", label="Training loss")

ax2.plot(epochs, val_loss, "b", label="Validation loss")

ax2.title.set_text("Training and Validation loss")



pyplot.legend(loc='best', shadow=True)

pyplot.tight_layout()

pyplot.show()
test_filenames = os.listdir(test_dir)

test_df = pd.DataFrame({

    'filename': test_filenames

})
nb_samples = test_df.shape[0]

nb_samples
test_df.head()
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(test_df, test_dir, x_col='filename',y_col=None,class_mode=None,

                                              batch_size=64,target_size=(200, 200),shuffle=False)
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/64))

threshold = 0.6

test_df['category'] = np.where(predict > threshold, 1,0)
sample_test = test_df.sample(n=9).reset_index()

sample_test.head()
pyplot.figure(figsize=(12, 12))

for index, row in sample_test.iterrows():

    filename = row['filename']

    category = row['category']

    img = load_img(test_dir + filename, target_size=(256, 256))

    pyplot.subplot(3, 3, index+1)

    pyplot.imshow(img)

    pyplot.xlabel(filename + '(' + "{}".format(category) + ')')

pyplot.tight_layout()

pyplot.show()
submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)
pyplot.figure(figsize=(10,5))

sns.countplot(submission_df['label'])

pyplot.title("(Test data)")