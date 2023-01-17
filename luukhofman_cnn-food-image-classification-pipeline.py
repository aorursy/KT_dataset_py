import os

import keras

import glob

import keras

import pandas as pd

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt



from PIL import Image

from sklearn.model_selection import train_test_split

from keras.models import load_model

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.models import Sequential, Model 

from keras import layers

from keras.layers import Conv2D,MaxPooling2D, BatchNormalization, Dropout, GlobalAveragePooling2D, Dense, Flatten

from keras import optimizers

from keras.applications.inception_v3 import InceptionV3

from keras.regularizers import l2
# Set general direction

dir = '/kaggle/input/food-recognition-challenge/'
# Create a dictionary from the  data file: {img.jpg : lable}



# Go to label directiory and tranfer labels to pd

labels = pd.read_csv(dir + 'train_labels.csv').set_index('img_name')



# Convert pd to dic -> {Image.jpg : label}

labels = labels.to_dict()

labels = labels[list(labels.keys())[0]]



print(f'lenght of dictionary: {len(labels)}')
# Create a PD dataframe with columns Images and Labels

# This DF is needed for the flow_from_dataframe method of the ImageDataGenerator class

df = pd.DataFrame(list(labels.items()))

df.columns = ['images', 'labels']

df = df.astype({'labels': str})



# Display the occurance of each class

df.groupby('labels').count()
train_df, validation_df = train_test_split(df, test_size=0.1)



print(f'train_df: {train_df.shape}')

print(f'validation_df: {validation_df.shape}')
# Preprocessing settings for training data

train_datagen = keras.preprocessing.image.ImageDataGenerator(

    rescale = 1./255,

    rotation_range=15,

    height_shift_range=0.1,

    width_shift_range=0.1,

    shear_range=0.01,

    zoom_range=[0.9, 1.25],

    brightness_range=[0.5, 1.5],

    horizontal_flip=True,

    vertical_flip=False,

    fill_mode='reflect')



# Preprocessing settings for test data

validation_datagen = keras.preprocessing.image.ImageDataGenerator(

    rescale = 1./255)
# Apply data transfers and create generators for training and test data

train_generator = train_datagen.flow_from_dataframe(

        dataframe=train_df,

        directory='/kaggle/input/food-recognition-challenge/train_set/train_set/',

        x_col="images",

        y_col="labels",

        target_size=(256, 256),

        batch_size=32,

        class_mode='categorical')



validation_generator = validation_datagen.flow_from_dataframe(

        dataframe=validation_df,

        directory='/kaggle/input/food-recognition-challenge/train_set/train_set/',

        x_col="images",

        y_col="labels",

        target_size=(256, 256),

        batch_size=32,

        class_mode='categorical')
# Cell below is used to initiate a cnn network



# Specify the shape of inputs

input_shape = (256,256,3)

batch_size = 32

n_classes = 80



# Set up model

base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=keras.layers.Input(shape=input_shape))

x = base_model.output

x = keras.layers.AveragePooling2D(pool_size=(6, 6))(x)

x = keras.layers.Dropout(.4)(x)

x = keras.layers.Flatten()(x)

predictions = keras.layers.Dense(80, init='glorot_uniform', W_regularizer=l2(.0005), activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)

# Display model

model.summary()



# Show if GPU is avialable

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Initaite mode, specify the optimizer, lossfunction and metrics

opt = keras.optimizers.SGD(lr=.01, momentum=.9)



model.compile(optimizer = opt,

              loss ='categorical_crossentropy',

              metrics = ['accuracy'])



# Save weights if model improved

filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')



# Early stopping in case val_loss < min_delta for a specific number of runs 

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.03, patience=5, verbose=1, mode='auto', baseline=None, restore_best_weights=False)



# Set callbacks 

callbacks_list = [checkpoint, early_stopping]



# same model as whole

model.save("Inception_transfer_10epocs.h5")



# Fit data & train model

model.fit_generator(

        train_generator,

        steps_per_epoch = train_generator.samples // batch_size,

        callbacks = callbacks_list,

        epochs = 1,

        validation_data = validation_generator,

        validation_steps = validation_generator.samples // batch_size)
# same model as whole

model.save("Inception_transfer_10_epocs.h5")
# Predict & submit 

# Uncomment to load previous model

# model = tf.keras.models.load_model('/kaggle/input/ml-model-v2/whole_model_v4.h5')



# Test data generator -> Rescale image size

test_datagen = keras.preprocessing.image.ImageDataGenerator(

    rescale = 1./255)



# apply test_datagen to input files

test_generator = test_datagen.flow_from_directory(

        '/kaggle/input/food-recognition-challenge/test_set',

        target_size=(256, 256),

        shuffle = False,

        class_mode=None,

        batch_size=1)



# Get the filenames & remove directory specification in front of filename

filenames = [filename[9:] for filename in test_generator.filenames]



# Not predicting in batches but each inidividual item, therefore we need to know the amount of predictions

nb_samples = len(filenames)



# Make predictions, returns probabilities for each class

print(f'Making predictions....')

predictions = model.predict_generator(test_generator,steps = nb_samples, verbose=1)



# Assign prediction to class with highest probability

y_pred_labels = np.argmax(predictions, axis = 1)



# Map predictions to the correct labels

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in y_pred_labels]



# Submit file

submission = pd.DataFrame({'img_name':filenames,'label':predictions})

submission.to_csv('submission.csv', index=False)

print("Done!")