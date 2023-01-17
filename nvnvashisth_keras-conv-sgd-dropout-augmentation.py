# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import zipfile

from os import makedirs

from os import listdir

from shutil import copyfile

from random import seed

from random import random

import shutil

if(os.path.isdir('/kaggle/working/train/')):

    shutil.rmtree('/kaggle/working/train/')

    shutil.rmtree('/kaggle/working/validation/')

    shutil.rmtree('/kaggle/working/training/')

    shutil.rmtree('/kaggle/working/test1/')

else:

    # create directories

    with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/train.zip","r") as z:

            z.extractall(".")

    dataset_home = '/kaggle/working/'

    subdirs = ['training/', 'validation/']

    for subdir in subdirs:

        labeldirs = ['dogs/', 'cats/']

        for labldir in labeldirs:

            newdir = dataset_home + subdir + labldir

            makedirs(newdir, exist_ok=True)

    # seed random number generator

    seed(1)

    # # define ratio of pictures to use for validation

    val_ratio = 0.25

    # # copy training dataset images into subdirectories

    src_directory = '/kaggle/working/train/'

    for file in listdir(src_directory):

        src = src_directory + '/' + file

        dst_dir = 'training/'

        if random() < val_ratio:

            dst_dir = 'validation/'

        if file.startswith('cat'):

            dst = dataset_home + dst_dir + 'cats/'  + file

            copyfile(src, dst)

        elif file.startswith('dog'):

            dst = dataset_home + dst_dir + 'dogs/'  + file

            copyfile(src, dst)
# plot dog photos from the dogs vs cats dataset

from matplotlib import pyplot

from matplotlib.image import imread

filename = '/kaggle/working/train/cat.0.jpg'

# filename = folder + file

image = imread(filename)

pyplot.imshow(image)

pyplot.show()
import tensorflow as tf

print(tf.test.gpu_device_name())

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True
# baseline model for the dogs vs cats dataset

import sys

from matplotlib import pyplot

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Dense

from keras.layers import Flatten

from keras.regularizers import l2

from keras.optimizers import SGD

from keras.callbacks import EarlyStopping

from keras.preprocessing.image import ImageDataGenerator



class cnn_arch():

	def __init__(self):

		self.input_size = (200,200,3)

		self.kernel_initializer = 'he_uniform'

		self.activation = 'relu'

		self.padding = 'same'

		self.loss = 'binary_crossentropy'

		self.kernel_regularizer = 'l2'

    

	# define cnn model

	def define_model(self):

		model = Sequential()

		model.add(Conv2D(32, (3, 3), 

                         activation=self.activation, 

                         kernel_initializer=self.kernel_initializer,

                         kernel_regularizer=l2(0.001),

                         padding=self.padding, input_shape=self.input_size))

		model.add(MaxPooling2D((2, 2)))

		model.add(Conv2D(64, (3, 3), 

                         activation=self.activation, 

                         kernel_initializer=self.kernel_initializer, 

                         kernel_regularizer=l2(0.001),

                         padding=self.padding))

		model.add(MaxPooling2D((2, 2)))

		model.add(Conv2D(128, (3, 3), 

                         activation=self.activation, 

                         kernel_initializer=self.kernel_initializer, 

                         kernel_regularizer=l2(0.001),

                         padding=self.padding))

		model.add(MaxPooling2D((2, 2)))

		model.add(Flatten())

		model.add(Dense(128, 

                        activation=self.activation, 

                        kernel_initializer=self.kernel_initializer,

                        kernel_regularizer=l2(0.001)))

		model.add(Dense(1, activation='sigmoid'))

		# compile model

		opt = SGD(lr=0.001, momentum=0.9)

		model.compile(optimizer='adam', loss=self.loss, metrics=['accuracy'])

		return model



	# plot diagnostic learning curves

	def summarize_diagnostics(self, history):

		# plot loss

		pyplot.subplot(211)

		pyplot.title('Cross Entropy Loss')

		pyplot.plot(history.history['loss'], color='blue', label='train')

		pyplot.plot(history.history['val_loss'], color='orange', label='test')

		# plot accuracy

		pyplot.subplot(212)

		pyplot.title('Classification Accuracy')

		pyplot.plot(history.history['accuracy'], color='blue', label='train')

		pyplot.plot(history.history['val_accuracy'], color='orange', label='test')

		# save plot to file

		filename = sys.argv[0].split('/')[-1]

		pyplot.savefig(filename + '_plot.png')

		pyplot.close()



	# run the test harness for evaluating a model

	def train(self):

		tf.debugging.set_log_device_placement(True)

		# define model

		model = self.define_model()

		# create data generators

		train_datagen = ImageDataGenerator(rescale=1.0/255.0,

			width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

		test_datagen = ImageDataGenerator(rescale=1.0/255.0)

		# early stopping

# 		es = EarlyStopping(monitor='val_loss', patience=20, verbose=2)

		# prepare iterators

		train_it = train_datagen.flow_from_directory('/kaggle/working/training/',

			class_mode='binary', batch_size=64, target_size=(200, 200))

		test_it = test_datagen.flow_from_directory('/kaggle/working/validation/',

			class_mode='binary', batch_size=64, target_size=(200, 200))

		# fit model

		history = model.fit_generator(train_it, steps_per_epoch=len(train_it),

                                      validation_data=test_it, validation_steps=len(test_it), epochs=100, verbose=2)

		# evaluate model

		_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=2)

		print('> %.3f' % (acc * 100.0))

		# model save

		model.save('final_model.h5')

		# learning curves

		self.summarize_diagnostics(history)





# entry point, run the test harness

cnn = cnn_arch().train()
image = imread('/kaggle/working/ipykernel_launcher.py_plot.png')

pyplot.imshow(image)

pyplot.show()
# make a prediction for a new image.

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.models import load_model

from os import listdir



with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/test1.zip","r") as z:

            z.extractall(".")

# load and prepare the image

def load_image(filename):

	# load the image

	img = load_img(filename, target_size=(200, 200))

	# convert to array

	img = img_to_array(img)

	# reshape into a single sample with 3 channels

	img = img.reshape(1, 200, 200, 3)

	return img

 

# load an image and predict the class

def run_example():

	model = load_model('final_model.h5')

	submission_df = pd.DataFrame(columns=['id','label'])

	# load the image

	src_directory = '/kaggle/working/test1/'

	for file in listdir(src_directory):

		img = load_image(src_directory+file)

		# predict the class

		result = model.predict(img)

		submission_df.loc[len(submission_df)] = [file.split('.')[0], result[0][0]]

	submission_df.to_csv('submission_070620201.csv', index=False)

 

# entry point, run the example

run_example()