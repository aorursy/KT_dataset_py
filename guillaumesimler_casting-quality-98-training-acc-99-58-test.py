# Import the standard libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Define the constance



FOLDER_ = 'Casting CNN'

BATCH_SIZE_ = 16

COLOR_SPECTRUM_ = (1)          # 1 if B&W, 3 if color

IMG_SIZE_ = (300, 300)
# import Keras Modules

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout
# Initialize the CNN



classifier = Sequential()
# Adding the layers

input_shape_ = (300,300) + (COLOR_SPECTRUM_, )

classifier.add(Conv2D(BATCH_SIZE_, (3,3), input_shape=input_shape_, activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))



# Add a second layer

classifier.add(Conv2D(BATCH_SIZE_, (3,3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))



# Flattening

classifier.add(Flatten())
# ANN Layer



# Add a second hidden layer

classifier.add(Dense(units = 128, activation='relu'))

classifier.add(Dropout(rate=.2))



# Add a second hidden layer

classifier.add(Dense(units = 128, activation='relu'))

classifier.add(Dropout(rate=.2))



# Add a third hidden layer

classifier.add(Dense(units = 64, activation='relu'))

classifier.add(Dropout(rate=.2))



classifier.add(Dense(units = 1, activation='sigmoid'))
classifier.compile(optimizer= 'adam',

                   loss='binary_crossentropy',

                   metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        vertical_flip=True,

        rotation_range=25)



test_datagen = ImageDataGenerator(rescale=1./255)
from pathlib import Path



dataset_folder =  Path('/kaggle/input/real-life-industrial-dataset-of-casting-product/casting_data')



if dataset_folder.exists():

    print(f'[-------]\nConnect the dataset folder at \n\t{str(dataset_folder)}\n[-------]')

else:

    print(f'[*******]\nConnecting the dataset folder failed \n[*******]')

training_files = dataset_folder / 'train/'

test_files = dataset_folder / 'test/'



training_set= train_datagen.flow_from_directory(

        training_files,

        target_size=(300,300),

        batch_size=BATCH_SIZE_,

        class_mode='binary',

        color_mode="grayscale")



test_set = test_datagen.flow_from_directory(

        test_files,

        target_size= (300,300),

        batch_size=BATCH_SIZE_,

        class_mode='binary',

        color_mode="grayscale")
training_size = len(training_set)

test_size = len(test_set)



print(f'{training_size} * {BATCH_SIZE_} = {training_size * BATCH_SIZE_}')

print(f'{test_size} * {BATCH_SIZE_} = {test_size * BATCH_SIZE_}')




# get the number of CPU threads



import multiprocessing

import tensorflow as tf



def set_workers(local = False):

    

    catcha =''

    workers = multiprocessing.cpu_count()

    

    if local:

        workers -= 1 

        catcha = 'locally '

        

    gpus = tf.config.experimental.list_physical_devices('GPU')



    print(f"Working with {workers} CPU threads {catcha}and with {len(gpus)} GPU" )

    

    return workers



workers_ = set_workers()

classifier.fit_generator(

        training_set,

        steps_per_epoch=training_size,

        epochs=25,

        validation_data=test_set,

        validation_steps=test_size,

        use_multiprocessing=True,

        workers=workers_)
classifier.summary()
import seaborn as sns



plt.rcParams["figure.figsize"] = (16,9)

sns.set(style="darkgrid")



data = pd.DataFrame(classifier.history.history)



plt.title('Visualisation of the training')

plt.ylabel('Loss/Accuracy')

plt.xlabel('# Epochs')

sns.lineplot(data=data, linewidth=3.5)
from sklearn.metrics import classification_report, confusion_matrix



test_set.reset

y_pred = classifier.predict_generator(generator = test_set, 

                                      steps = test_size,

                                      use_multiprocessing=True,

                                        workers=workers_)



y_pred = y_pred >= 0.5



print("[------]\nConfusion Matrix")

print(confusion_matrix(test_set.classes[test_set.index_array], y_pred))

print("[------]")
target_names = ['Defective parts', 'Good parts']



print('[------]\nClassification Report')

print(classification_report(test_set.classes[test_set.index_array], y_pred, target_names=target_names))

print("[------]")