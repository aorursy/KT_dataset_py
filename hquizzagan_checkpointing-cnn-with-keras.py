import numpy as np

import os

from os import listdir

import pandas as pd



import tensorflow as tf

import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import Activation, Dense

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import categorical_crossentropy

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow_hub as hub



from matplotlib import pyplot as plt

%matplotlib inline

from sklearn.metrics import confusion_matrix



import itertools

import seaborn as sns

from tensorflow import keras





from PIL import Image

from tqdm import tqdm



print('Setup completed!')
!pip install -q -U tf-hub-nightly

!pip install -q tfds-nightly

print('Tensorflow Hub requirements successfully installed!')
train_path = '../input/blood-cells/dataset2-master/dataset2-master/images/TRAIN/'

test_path = '../input/blood-cells/dataset2-master/dataset2-master/images/TEST/'



print('Paths ready!')
num_classes = len(listdir(train_path))

num_classes
# Pre-defined functions

def key_extractor(dictionary, value):

    '''

    Input:

    - Dictionary of any key,value pair

    - value to extract

    

    Return:

    - key of that value

    

    Example: dict = {'a':4, 'b':6, 'y':9,'z':3}

    key_extractor(dict, 3) => 'z'

    

    Caveat: Works only if all values are unique!

    '''

    for k,v in dictionary.items():

        if value == v:

            return k
class_dirs = [(train_path + '/' + category) for category in listdir(train_path)]

class_dirs
num_imgs_per_class = [len(listdir(class_dir)) for class_dir in class_dirs]
plt.figure(figsize=(5,5))

plt.title('Number of Images per Class')

sns.barplot(x=listdir(train_path), y=num_imgs_per_class)

plt.ylabel('Number of images per class')
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Tensorflow's Keras has an API that already handles converting RAW images into their array form

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,

                                   validation_split=0.2)



print('Ready to generate image data!')
image_size = 224

batch_size = 32



train_generator = data_generator.flow_from_directory(train_path,

                                                    target_size=(image_size, image_size),

                                                    class_mode='categorical',

                                                    batch_size=batch_size,

                                                    subset='training')



valid_generator = data_generator.flow_from_directory(train_path,

                                                    target_size=(image_size, image_size),

                                                    class_mode='categorical',

                                                    batch_size=batch_size,

                                                    subset='validation')



# I turned on the shuffle=False for convenience later when I need to extract the associated filename for the

# predicted classes

test_generator = data_generator.flow_from_directory(test_path,

                                                    target_size=(image_size, image_size),

                                                    class_mode='categorical',

                                                    shuffle=False,

                                                    batch_size=batch_size)
for image_batch, label_batch in train_generator:

    print("Image batch shape: ", image_batch.shape)

    print("Label batch shape: ", label_batch.shape)

    break
train_generator.class_indices
plt.figure(figsize=(20,8))

plt.subplots_adjust(hspace=0.5)

show_num_images = train_generator.batch_size

row = 3

col = np.ceil(show_num_images/row)



for i in range(show_num_images):

    plt.subplot(row,col,i+1)

    plt.imshow(image_batch[i])

#     plt.title(label_batch[i])

    plt.title(key_extractor(train_generator.class_indices, np.argmax(label_batch[i])))

    plt.axis('off')

_ = plt.suptitle("One Batch of Training Images (Labeled Accordingly)")
classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"



IMAGE_SHAPE = (224, 224)



classifier = tf.keras.Sequential([

    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))

])



print("TF Hub's classifier successfully loaded!")
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"



# Create the feature extractor

feature_extractor_layer = hub.KerasLayer(feature_extractor_url,

                                         input_shape=(224,224,3))



# Apply the feature_extractor on the first batch of images generated (for trial purposes only)

feature_batch = feature_extractor_layer(image_batch)

print(feature_batch.shape)
feature_extractor_layer.trainable = False

print('Feature extraction layer frozen!')
model = tf.keras.Sequential([

  feature_extractor_layer,

  layers.Dense(train_generator.num_classes)

])



model.summary()
# Use compile to configure the training process:

model.compile(optimizer=tf.keras.optimizers.Adam(),

                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

                metrics=['acc'])



print('Model compiled! \nReady for training!')
!pip install h5py

print('Ready to save models in the h5 format.')
# from tf.keras.callbacks import ModelCheckPoint

# filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"



checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.hdf5',

                                                monitor='val_loss',

                                                verbose=1,

                                                save_best_only=True,

                                                mode='min',

                                                period=1)

    

print('Callback successfully created!')
steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size)

valid_steps_per_epoch = np.ceil(valid_generator.samples/valid_generator.batch_size)



history = model.fit(train_generator,

                      steps_per_epoch=steps_per_epoch,

                      validation_data=valid_generator,

                      validation_steps=valid_steps_per_epoch,

                      epochs=3,

                      callbacks = [checkpoint],

                      verbose=2)



print('Model trained successfully!')