# Import Libraries

#%tensorflow_version 2.x

import tensorflow as tf

import glob

import os, shutil # Library for navigating files

from os import listdir, makedirs

from os.path import join, exists, expanduser



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.offsetbox import OffsetImage, AnnotationBbox



import math

from tensorflow.keras.preprocessing.image import ImageDataGenerator # Library for data augmentation

from keras.models import Sequential, Model

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.layers import Input, Concatenate, Activation, Dense, Flatten, Dropout, GlobalAveragePooling2D

from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras import backend, regularizers

from tensorflow.keras.utils import plot_model

from keras import applications

from keras import optimizers

from tensorflow.keras import backend as K

from tensorflow.keras import backend, models, layers, optimizers, regularizers

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

from IPython.display import display # Library to help view images

from PIL import Image # Library to help view images

np.random.seed(42)
#Training and testing images setup

train_data_dir = '../input/fruits/fruits-360/Training'

test_data_dir = '../input/fruits/fruits-360/Test'

#Some variables to use throughout the project

num_classes = 120

validsplit = .2

batchs = 128

img_row_col = 32


train_datagen = ImageDataGenerator(rescale=1./255

                                  ,rotation_range=15          # Rotate the images randomly by 15 degrees

                                  ,width_shift_range=0.2      # Shift the image horizontally by 20%

                                  ,height_shift_range=0.2     # Shift the image veritcally by 20%

                                  ,zoom_range=0.2             # Zoom in on image by 20%

                                  ,horizontal_flip=True       # Flip image horizontally 

                                  ,fill_mode='nearest'        # How to fill missing pixels after a augmentaion opperation

                                  ,validation_split=validsplit) # Split the data for use in a validation generator later



test_datagen = ImageDataGenerator(rescale=1./255)



# Next, create the flow_from_directory objects, that will be passed to the model fitting functions and

# the model testing function later on.

# Notice the "subset" parameter usage during the training and validation's flow_from_directory function,

# being used to specify the validation or training data split.

train_generator = train_datagen.flow_from_directory(train_data_dir

                                                   ,target_size=(img_row_col, img_row_col)

                                                   ,batch_size=batchs

                                                   ,class_mode='categorical'

                                                   ,shuffle=True

                                                   ,subset='training')



validation_generator = train_datagen.flow_from_directory(train_data_dir

                                                        ,target_size=(img_row_col, img_row_col)

                                                        ,batch_size=batchs

                                                        ,class_mode='categorical'

                                                        ,shuffle=True

                                                        ,subset='validation')



test_generator = test_datagen.flow_from_directory(test_data_dir

                                                 ,target_size=(img_row_col, img_row_col)

                                                 ,batch_size=batchs

                                                 ,class_mode='categorical'

                                                 ,shuffle=False)



backend.clear_session()



from keras import layers



x = Input(shape=(32,32,3)) 

# reduce overfitting

weight_decay = 1e-4



model = layers.Conv2D(filters = 16, kernel_size = 2,padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)

model = layers.BatchNormalization()(model)

model = layers.Conv2D(filters = 16, kernel_size = 2,padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)

model = layers.BatchNormalization()(model)

model = layers.MaxPooling2D(pool_size=2)(model)

model = layers.Dropout(0.2)(model)



model = layers.Conv2D(filters = 32, kernel_size = 2,padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)

model = layers.BatchNormalization()(model)

model = layers.Conv2D(filters = 32, kernel_size = 2,padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)

model = layers.BatchNormalization()(model)

model = layers.MaxPooling2D(pool_size=2)(model)

model = layers.Dropout(0.3)(model)



model = layers.Conv2D(filters = 64, kernel_size = 2,padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)

model = layers.BatchNormalization()(model)

model = layers.Conv2D(filters = 64, kernel_size = 2,padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)

model = layers.BatchNormalization()(model)

model = layers.MaxPooling2D(pool_size=2)(model)

model = layers.Dropout(0.4)(model)



model = layers.Conv2D(filters = 128, kernel_size = 2,padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)

model = layers.BatchNormalization()(model)

model = layers.Conv2D(filters = 128, kernel_size = 2,padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)

model = layers.BatchNormalization()(model)

model = layers.MaxPooling2D(pool_size=2)(model)

model = layers.Dropout(0.5)(model)



model = Flatten()(model)

model = Dense(num_classes,activation = 'softmax')(model)

model1 = Model(inputs=x, outputs=model)



plot_model(model1,show_shapes=True,expand_nested=True)
from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)
training_data = pd.DataFrame(train_generator.classes, columns=['classes'])

testing_data = pd.DataFrame(validation_generator.classes, columns=['classes'])
def create_stack_bar_data(col, df):

    aggregated = df[col].value_counts().sort_index()

    x_values = aggregated.index.tolist()

    y_values = aggregated.values.tolist()

    return x_values, y_values
x1, y1 = create_stack_bar_data('classes', training_data)

x1 = list(train_generator.class_indices.keys())



trace1 = go.Bar(x=x1, y=y1, opacity=0.75, name="Class Count")

layout = dict(height=400, width=1200, title='Class Distribution in Training Data', legend=dict(orientation="h"), 

                yaxis = dict(title = 'Class Count'))

fig = go.Figure(data=[trace1], layout=layout);

iplot(fig);
x1, y1 = create_stack_bar_data('classes', testing_data)

x1 = list(train_generator.class_indices.keys())



trace1 = go.Bar(x=x1, y=y1, opacity=0.75, name="Class Count")

layout = dict(height=400, width=1200, title='Class Distribution in Testing Data', legend=dict(orientation="h"), 

                yaxis = dict(title = 'Class Count'))

fig = go.Figure(data=[trace1], layout=layout);

iplot(fig);
# Counts breakouts of the images, and capture the values in variables, for utilize later on.

train_image_cnt = train_generator.n

val_image_cnt = validation_generator.n

test_image_cnt = test_generator.n

print('train generator count: ',

       train_image_cnt,

       ', validation generator count: ',

       val_image_cnt,

       ', test generator count: ',

       test_image_cnt)
# Show/plot sample images with labels

width = 10

height = 10

fig=plt.figure(figsize=(10, 10))

columns = 3

rows = 3

x_batch, y_batch = next(test_generator)

for i in range(1, columns*rows +1):

    image = x_batch[np.random.randint(1,batchs-1)]

    fig.add_subplot(rows, columns, i)

    plt.imshow(image)

plt.show()
def lr_schedule(epoch):

   initial_lrate = 0.1

   drop = 0.5

   epochs_drop = 10.0

   learningrate = initial_lrate * math.pow(drop,  

           math.floor((1+epoch)/epochs_drop))

   return learningrate



epochs = 80

# show progress bar

verbs = 1



model1.compile(loss='categorical_crossentropy',

                  optimizer='rmsprop',

                  metrics=['accuracy'])





history = model1.fit_generator(train_generator,

                                  steps_per_epoch=train_image_cnt//(batchs*3),

                                  epochs=epochs,

                                  callbacks=[LearningRateScheduler(lr_schedule)],

                                  validation_data = validation_generator,

                                  validation_steps = validation_generator.samples//(batchs*3),

                                  verbose=verbs)
test_loss, test_acc = model1.evaluate_generator(test_generator, steps = 50)



print('Test Accuracy:', test_acc)

      
history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

acc_values = history_dict['accuracy']

val_acc_values = history_dict['val_accuracy']

epochs = range(1, len(history_dict['accuracy']) + 1)



plt.plot(epochs, loss_values, 'bo', label = 'Training loss')

plt.plot(epochs, val_loss_values, 'red', label = 'Validation loss')

plt.title('Training & validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.grid()

plt.show()



plt.plot(epochs, acc_values, 'bo', label = 'Training accuracy')

plt.plot(epochs, val_acc_values, 'green', label = 'Validation accuracy')

plt.title('Training & validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.grid()

plt.show()

# Get label mapping of class and label number

test_generator = test_datagen.flow_from_directory(train_data_dir, target_size=(img_row_col, img_row_col))

print(test_generator.class_indices)

labels = [None] * len(test_generator.class_indices)

for k, v in test_generator.class_indices.items():

  labels[v] = k
#!/bin/bash
# Visualizing predictions

result = np.round(model1.predict_generator(validation_generator))

import random

test_files = []

actual_res = []

test_res = []

for i in range(0, 4):

  rng = random.randint(0, len(validation_generator.filenames))

  test_files.append(test_data_dir +  validation_generator.filenames[rng])

  actual_res.append(validation_generator.filenames[rng].split('../input/fruits/fruits-360/')[0])

  test_res.append(labels[np.argmax(result[rng])])



from IPython.display import Image, display

for i in range(0, 4):

  #![title]display(Image(test_generator[i]))       

   print("Actual class: " + str(actual_res[i]))

   print("Predicted class: " + str(test_res[i]))
