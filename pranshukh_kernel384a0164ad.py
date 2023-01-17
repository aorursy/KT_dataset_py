!ls ../input/data
#Creating train and validation data

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

# define path for train dir 

train_dir = "../input/data"



train_datagen = ImageDataGenerator( 

    rescale=1. / 255, 

    shear_range=0.2, 

    zoom_range=0.2, 

    horizontal_flip=True,

    validation_split=0.2) 

  

train_generator = train_datagen.flow_from_directory( 

    train_dir, 

    target_size=(256, 256), 

    batch_size=32,

    class_mode='categorical',

    subset='training') 



validation_generator = train_datagen.flow_from_directory(

    train_dir, 

    target_size=(256,256),

    batch_size=32,

    class_mode='categorical',

    subset='validation')

#Creating Classes

classes = ["Airplane" , "Airport" , "BaseBall" , "Basketball" , "Beach" , "Bridge" , "Chappral" , "Church" , "Farmland" , "Commercial Area" , "Residential" , "Desert" , "Forest" , "Freeway" , "Golf" , "Track" , "Harbor" , "Industrial" , "Intersection" , "Island" , "Lake" ,"Meadow" , "Medium Residential" , "Home Park" , "Mountain" , "Overpass" , "Parking" , "railway" , "farmland" , "Roundabout" , "Runway"]

print(len(classes))
#Visualizing and understanding data

import matplotlib.pyplot as plt

import numpy as np



batch = train_generator[31]

images = batch[0]

labels = batch[1]



for i in range(32):

  image = images[i]

  label = labels[i]

  plt.figure()

  plt.imshow(image)

  for i in range(31):

    if label[i] == 1:

      name = classes[i]

      break

  plt.text(100 , 300 , name)
import keras

from keras.models import Model, Sequential, load_model

from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout

from keras.optimizers import Adam

from keras import applications

from keras.layers import Flatten, Dense, Input
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1', 

                 input_shape=(256, 256, 3)))

model.add(MaxPooling2D((2, 2), name='maxpool_1'))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'))

model.add(MaxPooling2D((2, 2), name='maxpool_2'))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3'))

model.add(MaxPooling2D((2, 2), name='maxpool_3'))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4'))

model.add(MaxPooling2D((2, 2), name='maxpool_4'))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(512, activation='relu', name='dense_1'))

model.add(Dense(256, activation='relu', name='dense_2'))

model.add(Dense(31, activation='sigmoid', name='output'))



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit_generator(train_generator, steps_per_epoch= 1209//32, epochs=25 ,validation_data = validation_generator, 

    validation_steps = 279//32,)
model.save("modelkag.h5")
score = model.evaluate_generator(train_generator)

print("Training accuracy Score: ", score[1]*100)

val_score = model.evaluate_generator(validation_generator)

print("Validation accuracy Score: ", val_score[1]*100)



print(score)

print(val_score)