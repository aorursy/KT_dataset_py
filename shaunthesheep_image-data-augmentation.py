! rm -rf /kaggle/working/*
import random

from shutil import copyfile



import os,sys

import zipfile

import shutil

from os import path, getcwd, chdir



## Bare minimum library requirement

import tensorflow as tf

import keras

#Keras provide API for Augmentation helps in generation

from tensorflow.keras.optimizers import RMSprop
! cp -R /kaggle/input/* /kaggle/working
#List down all directories in "/kaggle/input/"

for dirName,_,fileName in os.walk("/kaggle/input/microsoft-catsvsdogs-dataset/"):

    print(dirName)
#List down all directories in "/kaggle/working/"

for dirName,_,fileName in os.walk("/kaggle/working/microsoft-catsvsdogs-dataset/"):

    count = 0

    print("Directory:: ",dirName)
! mkdir /kaggle/working/microsoft-catsvsdogs-dataset/training/

! mkdir /kaggle/working/microsoft-catsvsdogs-dataset/training/Dog/

! mkdir /kaggle/working/microsoft-catsvsdogs-dataset/training/Cat/



! mkdir /kaggle/working/microsoft-catsvsdogs-dataset/testing/

! mkdir /kaggle/working/microsoft-catsvsdogs-dataset/testing/Dog/

! mkdir /kaggle/working/microsoft-catsvsdogs-dataset/testing/Cat/
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE,DESTINATION):

    files = []

    for filename in os.listdir(SOURCE):

        file = SOURCE + filename

        if os.path.getsize(file) > 0:

            files.append(filename)

        else:

            print(filename + " has not enough pixels to represent it as an image, seems corrupted so ignoring.")



    training_length = int(len(files) * SPLIT_SIZE)

    testing_length = int(len(files) - training_length)

    shuffled_set = random.sample(files, len(files))

    training_set = shuffled_set[0:training_length]

    testing_set = shuffled_set[-testing_length:]



    for filename in training_set:

        this_file = SOURCE + filename

        destination = TRAINING + filename

        copyfile(this_file, destination)



    for filename in testing_set:

        this_file = SOURCE + filename

        destination = TESTING + filename

        copyfile(this_file, destination)



#####################################################################################



DESTINATION = "/kaggle/working"



CAT_SOURCE_DIR = "/kaggle/working/microsoft-catsvsdogs-dataset/PetImages/Cat/"

DOG_SOURCE_DIR = "/kaggle/working/microsoft-catsvsdogs-dataset/PetImages/Dog/"



TRAINING_CATS_DIR = "/kaggle/working/microsoft-catsvsdogs-dataset/training/Cat/"

TESTING_CATS_DIR = "/kaggle/working/microsoft-catsvsdogs-dataset/testing/Cat/"



TRAINING_DOGS_DIR = "/kaggle/working/microsoft-catsvsdogs-dataset/training/Dog/"

TESTING_DOGS_DIR = "/kaggle/working/microsoft-catsvsdogs-dataset/testing/Dog/"
split_size = .9

split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size,DESTINATION)

split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size,DESTINATION)
print("Total Cat iamge count :: ",len(os.listdir(TRAINING_CATS_DIR)))

print("Total Dog iamge count :: ",len(os.listdir(TRAINING_DOGS_DIR)))
%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from matplotlib.pyplot import imread, imshow, subplots, show

CAT_TRAINING_DIR , DOG_TRAINING_DIR  =  TRAINING_CATS_DIR,TRAINING_DOGS_DIR



# Parameters for our graph; we'll output images in a 4x4 configuration

nrows = 4

ncols = 4



# Index for iterating over images

pic_index = 0
try:

    # Set up matplotlib fig, and size it to fit 4x4 pics

    fig = plt.gcf()

    fig.set_size_inches(ncols * 4, nrows * 4)

    pic_index += 8



    next_cat_pix = [os.path.join(CAT_TRAINING_DIR, fname) for fname in os.listdir('/kaggle/working/microsoft-catsvsdogs-dataset/PetImages/Cat/')[pic_index - 8:pic_index]]

    next_dog_pix = [os.path.join(DOG_TRAINING_DIR, fname) for fname in os.listdir('/kaggle/working/microsoft-catsvsdogs-dataset/PetImages/Dog/')[pic_index - 8:pic_index]]



    for i, img_path in enumerate(next_cat_pix + next_dog_pix):

        # Set up subplot; subplot indices start at 1

        sp = plt.subplot(nrows, ncols, i + 1)

        sp.axis('On')  # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)

        plt.imshow(img)



    plt.show()



except:

    pass
%matplotlib inline

from tensorflow.keras.preprocessing.image import ImageDataGenerator



def plot(data_generator):

    """

    Plots 4 images generated by an object of the ImageDataGenerator class.

    """

    data_generator.fit(images)

    image_iterator = data_generator.flow(images)

    

    #Plot the images given by the iterator

    fig, rows = subplots(nrows=1, ncols=4, figsize=(18, 18))

    for row in rows:

        row.imshow(image_iterator.next()[0].astype('int'))

        row.axis('on')

    show()
def imageAugmentor():

    data_generator = ImageDataGenerator(rotation_range=180)

    plot(data_generator)



    data_generator = ImageDataGenerator(featurewise_center=False,

                                        width_shift_range=0.65)

    plot(data_generator)



    data_generator = ImageDataGenerator(featurewise_center=False,

                                        width_shift_range=0.65)

    plot(data_generator)



    data_generator = ImageDataGenerator(vertical_flip=True,

                                        zoom_range=[0.2, 0.9],

                                        width_shift_range=0.2)

    plot(data_generator)



    data_generator = ImageDataGenerator(horizontal_flip=True,

                                        zoom_range=[1, 1.5],

                                        width_shift_range=0.2)

    plot(data_generator)



    data_generator = ImageDataGenerator(width_shift_range=[0.1, 0.5])

    plot(data_generator)



    data_generator = ImageDataGenerator(zoom_range=[1, 2], rotation_range=260)

    plot(data_generator)
pic_index += 8

next_pic = [

    os.path.join(CAT_TRAINING_DIR, fname) for fname in os.listdir('/kaggle/input/microsoft-catsvsdogs-dataset/PetImages/Cat/')[pic_index - 8:pic_index]

]

image = plt.imread(next_pic[0])

# Creating a dataset which contains just one image.

images = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

imshow(images[0])

show()
imageAugmentor()
dict = {}

training_data_path = "/kaggle/working/microsoft-catsvsdogs-dataset/training/"

for directory in os.listdir(training_data_path):

    count = 0

    for fileName in os.listdir(training_data_path + directory):

        count += 1



    dict.update({"{0}".format(directory): count})

print(dict)
class NeuralNet:

    '''

    Responsible for Neural net skeleton

    '''

    '''

    Sequential design of layering to interconnect various layers.

    Hawk eye view would be

     ___________________________________________________

    |conv-->pool-->conv-->pool-->flatten-->dense-->dense|

     ---------------------------------------------------

    

    #Basic parameters to be passed on call 

    #1.training_data_path

    #2.validation_data_path

    #3.callback

    #4.epochs

    #5.batch_size

    #6.learning_rate

    '''

    

    def neuralModeling(self, training_data_path, validation_data_path,

                       callback, epochs, batch_size, learning_rate):

        model = tf.keras.models.Sequential([

            tf.keras.layers.Conv2D(16, (3, 3),

                                   activation='relu',

                                   input_shape=(150, 150, 3)),

            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(512, activation='relu'),

            tf.keras.layers.Dense(1, activation='sigmoid')

        ])



        #Model compilation

        model.compile(

            optimizer=RMSprop(lr=learning_rate),

            loss='binary_crossentropy',

            metrics=['accuracy']

        )



        #model summary

        model.summary()



        #Make datagen for Train generator

        train_datagen = ImageDataGenerator(rescale=1./255)



        #Train generator

        train_generator = train_datagen.flow_from_directory(

            training_data_path,

            target_size=(150, 150),

            batch_size=batch_size,

            class_mode='binary')

        

        #Make datagen for validation generator

        validation_datagen = ImageDataGenerator(rescale=1./255)



        #validation generator

        validation_generator = validation_datagen.flow_from_directory(

            validation_data_path,

            target_size=(150, 150),

            batch_size=batch_size,

            class_mode='binary')

        logdir = "/kaggle/working/logs" + datetime.now().strftime("%Y%m%d-%H%M%S")

        

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        

        history = model.fit(train_generator,

                            validation_data=validation_generator,

                            epochs=epochs,

                            verbose=1,

                            callbacks = [tensorboard_callback]

                            )



        return history, model



    '''

    Constructor of the class    

    '''

    

    def __init__(self):

        print("Object getting created")
from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



from datetime import datetime

from packaging import version



class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        

        if (type(logs.get('accuracy'))!= None and logs.get('accuracy') > 0.99):

            print(

                "\n\n\nGot accuracy above 0.99% so cancelling any further training! \n\nas it might cause Overfitting\n\n"

            )

            self.model.stop_training = True





callback = myCallback()
#Training data

training_data_path = "/kaggle/working/microsoft-catsvsdogs-dataset/training/"

validation_data_path = "/kaggle/working/microsoft-catsvsdogs-dataset/testing/"

#Epochs

epochs = 10

#Batch size

batch_size=100

#Learning Rate

learning_rate = 0.001
''' #Basic parameters to be passed on call 

    #1.training_data_path

    #2.validation_data_path

    #3.callback

    #4.epochs

    #5.batch_size

    #6.learning_rate

'''

net = NeuralNet()

history, model = net.neuralModeling(training_data_path, validation_data_path,callback, epochs, batch_size, learning_rate)
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



epochs = range(len(acc))

plt.figure(figsize=(17, 10))

plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(loc=0)

plt.show()
import matplotlib.pyplot as plt

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))

plt.figure(figsize=(17,10))

plt.plot(epochs, loss, 'r', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(loc=0)

plt.show()