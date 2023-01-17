#Loading the packages

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator
#Creating Model

model = Sequential([



        #Adding 5 Convolution Layers

        Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),

        MaxPooling2D(2,2),



        Conv2D(32, (3,3), activation='relu'),

        MaxPooling2D(2,2),



        Conv2D(64, (3,3), activation='relu'),

        MaxPooling2D(2,2),



        Conv2D(64, (3,3), activation='relu'),

        MaxPooling2D(2,2),



        Conv2D(64, (3,3), activation='relu'),

        MaxPooling2D(2,2),



        #Flattenning the Pooled Parameters

        Flatten(),



        #Creating Fully Connected Networks of 512 Neurons followed by 128 Neurons

        Dense(512, activation='relu'),

        Dense(128, activation='relu'),





        #Final Prediction is Binary: NORMAL or PNEUMONIA, so using single output neuron with sigmoid funtion

        #to give output in 0-1 where 0 for NORMAL and 1 for PNEUMONIA

        Dense(1, activation='sigmoid')

    ])



#Compiling the model

model.compile(

    optimizer = 'adam',

    loss = 'binary_crossentropy',

    metrics = [

        'accuracy',

        tf.keras.metrics.Precision(name='precision'),

        tf.keras.metrics.Recall(name='recall')

    ]

)
model.summary()
#Getting the dataset ready to train the model



#Rescaling image pixel values in 0-1

traindatagen = ImageDataGenerator(rescale = 1/255)

valdatagen = ImageDataGenerator(rescale = 1/255)





#The generators below takes the path to a directory & generates batches of augmented data.

#Target size will rescale the images with different shapes to the same shape



train_generator = traindatagen.flow_from_directory(

    '../input/chest-xray-pneumonia/chest_xray/train/',

    target_size = (300,300),

    batch_size = 64,

    class_mode = 'binary'

)



validation_generator = valdatagen.flow_from_directory(

    '../input/chest-xray-pneumonia/chest_xray/test/',

    target_size = (300, 300),

    batch_size = 64,

    class_mode = 'binary'

)
ncount = len([name for name in os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL')])

pcount = len([name for name in os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA')])



print('PNUEMONIA Images:', pcount)

print('NORMAL Images:', ncount)
#Setting the classweights to remove the imbalancement in the dataset

class_weight = {0: ((ncount+pcount)/ncount)/2 ,1: ((ncount+pcount)/pcount)/2}

print('Weight for Class NORMAL:', class_weight[0])

print('Weight for Class PNEUMONIA:', class_weight[1])
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("chest-xray-model.h5",

                                                    save_best_only=True)



early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,

                                                     restore_best_weights=True)



# This function keeps the learning rate at 0.001 for the first ten epochs

# and decreases it exponentially after that.

def scheduler(epoch):    

    if epoch < 10:

        return 0.001

    else:

        return 0.001 * tf.math.exp(0.1 * (10 - epoch))



lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)





callbacks = [checkpoint_cb, early_stopping_cb, lr_scheduler]
#Training the model

history = model.fit(

    train_generator,

    steps_per_epoch = 10,

    epochs = 20,

    validation_data = validation_generator,

    class_weight = class_weight,

    callbacks = callbacks

)
#Visualising Model's Performance



fig, ax = plt.subplots(1, 4, figsize=(20,3))

ax = ax.ravel()



for i, met in enumerate(['precision','recall','accuracy', 'loss']):

    ax[i].plot(history.history[met])

    ax[i].plot(history.history['val_' + met])

    ax[i].set_title('Model {}'.format(met))

    ax[i].set_xlabel('epochs')

    ax[i].set_ylabel(met)

    ax[i].legend(['train', 'val'])
#Loading the unseen data for evaluation

testdatagen = ImageDataGenerator(rescale = 1/255)



test_generator = valdatagen.flow_from_directory(

    '../input/chest-xray-pneumonia/chest_xray/test/',

    target_size = (300, 300),

    batch_size = 64, 

    class_mode = 'binary'

)
#Evaluating the trained model



result = model.evaluate_generator(test_generator, 624)



print("loss at eval data:", result[0])

print('accuracy at eval data:', result[1])

print("precision at eval data:", result[2])

print("recall at eval data:", result[3])
