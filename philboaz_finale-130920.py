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
from __future__ import print_function

import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, MaxPool2D, AveragePooling2D

import os



import numpy as np



import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt



from sklearn.metrics import confusion_matrix, classification_report

import itertools



%matplotlib inline

# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout,Flatten

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.callbacks import TensorBoard,EarlyStopping

%load_ext tensorboard



# Helper libraries

import numpy as np

import random

import matplotlib.pyplot as plt

import datetime



from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout

from keras.regularizers import l2

from keras.regularizers import l1

from tensorflow.keras import regularizers



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.callbacks import TensorBoard,EarlyStopping



kernel_regularizer=tf.keras.regularizers.l2(0.01)

import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory



print(tf.__version__)
import numpy as np

from keras import backend as K

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix



#Start

train_data_path = '../input/animal1209/animal_dataset_intermediate_new/train_split/train'

test_data_path = '../input/animal1209/animal_dataset_intermediate_new/train_split/val'

img_rows = 150

img_cols = 150

epochs = 100

batch_size = 32

num_of_train_samples = 5736

num_of_test_samples = 2460



#Image Generator

train_datagen = ImageDataGenerator(rescale=1. / 255,

                                   rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True,

                                   fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(train_data_path,

                                                    target_size=(img_rows, img_cols),

                                                    batch_size=batch_size,

                                                    class_mode='categorical')



validation_generator = test_datagen.flow_from_directory(test_data_path,

                                                        target_size=(img_rows, img_cols),

                                                        batch_size=batch_size,

                                                        class_mode='categorical')



train_generator_2 = image_dataset_from_directory(

    directory=r"../input/animal1209/animal_dataset_intermediate_new/train_split/train",

    labels = "inferred", label_mode = 'int',

    validation_split = 0.2,

    subset = "training",

    seed = 1337,

    image_size=(224, 224),

    batch_size=32

)



#visualizing the data

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

for images, labels in train_generator_2.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(int(labels[i]))

        plt.axis("off")
#Define optimizer 



STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

steps_per_epoch=STEP_SIZE_TRAIN 



lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(

  0.005, decay_steps=steps_per_epoch*1000,decay_rate=1,staircase=False)



optimizer_2 = SGD(lr_schedule)

optimizer= SGD(lr = 0.01)
# Define  Callbacks 



reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=5, min_lr=0.001)

callbacks=[reduce_lr]



earlystopping_callback = EarlyStopping(

    monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto',

    baseline=None, restore_best_weights=True)
#MLP model 62%/42%



model3 = Sequential()

#input layer size is 784 after flattening

model3.add(Flatten(input_shape=(224, 224, 3)))

  

#hidden layer with 512 neurons

model3.add(Dense(512, activation='relu',kernel_regularizer=regularizers.l1(0.01)))

model3.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l1(0.01)))

model3.add(Dropout(0.5))

model3.add(BatchNormalization())



model3.add(Dense(512, activation='relu',kernel_regularizer=regularizers.l1(0.01)))

model3.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l1(0.01)))

model3.add(Dropout(0.5))

model3.add(BatchNormalization())

model3.add(Dense(5, activation='softmax'))



model3.summary()





# compile model

model3.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['accuracy'])
# CNN model



model = Sequential()

model.add(Convolution2D(32, (3, 3), input_shape=(img_rows, img_cols, 3), padding='valid'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Convolution2D(32, (3, 3), padding='valid'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Convolution2D(64, (3, 3), padding='valid'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(5))

model.add(Activation('softmax'))



model.summary()



model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])
#Train

model.fit_generator(train_generator,

                    steps_per_epoch=num_of_train_samples // batch_size,

                    epochs=epochs,

                    validation_data=validation_generator,

                    validation_steps=num_of_test_samples // batch_size)



model.save_weights('model.h5')


# TensorFlow and tf.keras



%load_ext tensorboard

import datetime

print(tf.__version__)



# run the tensorboard command to view the visualizations.



log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

%tensorboard --logdir logs/fit
#Evaluate the model 

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size

model.evaluate(validation_generator,

steps=STEP_SIZE_VALID)


Y_pred = model.predict_generator(validation_generator, num_of_test_samples)

y_pred = np.argmax(Y_pred, axis=1)



y_pred[200]



y_pred.shape
predictions = [labels[i] for i in y_pred]

predictions


target_names = ['elefante_train', 'farfalla_train', 'mucca_train','pecora_train','scoiattolo_train']

print(classification_report(validation_generator.labels, y_pred, target_names=target_names))
#Extract the test data => I didnt find a way without creating a new folder on colab



test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.)



test_generator = test_datagen.flow_from_directory("../input/animal-ori/animal_dataset_intermediate",

                                                    

                                                    class_mode = 'categorical', 

                                                    target_size = (150, 150))
#prediction on test data 



num = 9106



Y_pred_test = model.predict_generator(test_generator)

y_pred_test = np.argmax(Y_pred_test, axis=1)

y_pred_test

Y_pred_test
y_final = y_pred_test[0:910]
print(*y_final, sep = ", ")  
res = pd.DataFrame(y_final) #preditcions are nothing but the final predictions of your model on input features of your new unseen test data

 # its important for comparison. Here "test_new" is your new test dataset

res.columns = ["prediction"]

res.to_csv("prediction_results.csv")      # the csv file will be saved locally on the same location where this notebook is located.
res
im = pd.read_csv('../input/animal-ori/animal_dataset_intermediate/Testing_set_animals.csv')

im



rei = pd.concat([im,res],  axis = 1) 
rei.drop('target', axis = 1)
rei['animal'] = rei['prediction'] 

rei
rei['animal'] =rei['animal'].replace(to_replace=2,value = "mucca") 
rei.head(100)
rei['animal'] =rei['animal'].replace(to_replace=3,value = "pecora") 


rei['animal'] =rei['animal'].replace(to_replace=4,value = "scoiattolo") 
rei['animal'] =rei['animal'].replace(to_replace=0,value = "elefante") 
rei
rei['animal'] =rei['animal'].replace(to_replace=1,value = "farfalla") 
rei.head(200)
rei.to_csv('final')
y_final.shape
#!pip install jovian --upgrade
#import jovian
#jovian.commit(project='Animal finale 2020')