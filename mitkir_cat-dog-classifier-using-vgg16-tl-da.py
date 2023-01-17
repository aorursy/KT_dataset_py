import os

import zipfile

import numpy as np

import pandas as pd

import random

import matplotlib.pyplot as plt

import tensorflow as tf

from keras.applications.vgg16 import VGG16

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split





from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input



from keras.layers import Input, Flatten, Dense

from keras.models import Model

import numpy as np
# Define some parameters and constants

FOLDER_PATH = "../input/dogs-vs-cats/train/train/"

IMAGE_SIZE = (224, 224)

EPOCHS = 5
# Create DataFrame to analyze dataset

filenames = os.listdir(FOLDER_PATH)

labels = []

for name in filenames:

    label = name.split('.')[0]

    if label == 'dog':

        labels.append(1)

    else:

        labels.append(0)



df = pd.DataFrame({

    'filename': filenames,

    'label': labels

})



df.head()
# Detect class balance for dataset

df['label'].value_counts().plot.bar()
# See sample images

sample_train = df.head(12)

sample_train.head()

plt.figure(figsize=(12, 24))

for index, row in sample_train.iterrows():

    filename = row['filename']

    category = row['label']

    img = load_img(FOLDER_PATH+filename, target_size=IMAGE_SIZE)

    plt.subplot(6, 3, index+1)

    plt.imshow(img)

    plt.xlabel(filename + '(' + "{}".format(category) + ')' )

plt.tight_layout()

plt.show()
from keras.applications import VGG16

# Load the VGG model with trained on ImageNet

vgg_imagenet = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))



# Freeze the layers except last 4 layers

for layer in vgg_imagenet.layers[:-4]:

    layer.trainable = False



# Show a summary of the model. Check the number of trainable parameters

vgg_imagenet.summary()
from keras import models

from keras import layers

from keras import optimizers



# Create new model (VGG ImageNet (CNN) + fully-connected layers)

model = models.Sequential()

 

# Add the VGG ImageNet model

model.add(vgg_imagenet)

 

# Add new FC layers

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(2, activation='softmax'))

 

# Show a summary of the model. Check the number of trainable parameters

model.summary()
# Split dataset to validation and train

df['label'] = df['label'].replace({0:'cat', 1:'dog'})



train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
# Check train dataset

train_df.shape
# Check validation dataset

validate_df.shape
train_datagen = ImageDataGenerator(

      rescale=1./255,

      rotation_range=20,

      width_shift_range=0.2,

      height_shift_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest')

 

validate_datagen = ImageDataGenerator(rescale=1./255)

 

# Define the batchsize for train and validation datasets

train_batchsize = 50

validate_batchsize = 50





# Create data generators

train_generator = train_datagen.flow_from_dataframe(train_df, 

                                                    FOLDER_PATH, 

                                                    x_col = 'filename', 

                                                    y_col = 'label', 

                                                    batch_size=train_batchsize, 

                                                    class_mode='categorical', 

                                                    target_size=IMAGE_SIZE)



validate_generator = validate_datagen.flow_from_dataframe(validate_df, 

                                                          FOLDER_PATH, 

                                                          x_col = 'filename', 

                                                          y_col = 'label', 

                                                          batch_size=validate_batchsize, 

                                                          class_mode='categorical', 

                                                          target_size=IMAGE_SIZE)
# Compile the model

model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['accuracy'])



# Train the model

history = model.fit_generator(

      train_generator,

      steps_per_epoch=train_generator.samples/train_generator.batch_size,

      epochs=EPOCHS,

      validation_data=validate_generator,

      validation_steps=validate_generator.samples/validate_generator.batch_size,

      verbose=1)
model.save_weights("vgg16_weights.h5")
model.save('vgg16.h5')
model_json = model.to_json()

with open("vgg16.json", "w") as json_file:

    json.dump(model_json, json_file)
# Let us see the loss and accuracy curves

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

 

epochs = range(len(acc))

 

plt.plot(epochs, acc, 'b', label='Training acc')

plt.plot(epochs, val_acc, 'r', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

 

plt.figure()

 

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

 

plt.show()
submission = pd.read_csv("../input/dogs-vs-cats/sampleSubmission.csv")



print(submission["label"][0])

pd.options.mode.chained_assignment = None  # default='warn'



for e,i in enumerate(os.listdir("../input/dogs-vs-cats/test1/test1")):

    print(i)

    output=[]

    img = image.load_img(os.path.join("../input/dogs-vs-cats/test1/test1",i), target_size=IMAGE_SIZE)

    img = np.asarray(img)

    img = np.expand_dims(img, axis=0)

    output = model.predict(img)

    if output[0][0] > output[0][1]:

#         print("cat")

        submission["id"][e]=i

        submission["label"][e]="cat"

    else:

#         print('dog')

        submission["id"][e]=i

        submission["label"][e]="dog"
submission.to_csv("my_submission.csv", index = False)