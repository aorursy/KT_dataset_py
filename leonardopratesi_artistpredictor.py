# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#let's see the artists

artists = pd.read_csv('../input/best-artworks-of-all-time/artists.csv')

artists.head()
#Processing Data

artists = artists.sort_values(by=['paintings'], ascending=False)

#consider only painter with at least 200 paintings

artists_top = artists[artists['paintings'] >= 200]

artists_top = artists_top[['name', 'paintings']]



print(artists_top)



#Correct error Durer

updated_name = "Albrecht_Dürer".replace("_", " ")

artists_top.iloc[4, 0] = updated_name
#print some paintings

images_dir = '../input/best-artworks-of-all-time/images/images'

artists_dirs = os.listdir(images_dir)

artists_top_name = artists_top['name'].str.replace(' ', '_').values

n= 10

fig, axes = plt.subplots(1, n, figsize=(20,20))



for i in range(n):

    random_artist = random.choice(artists_top['name'].str.replace(' ', '_').values)

    random_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))

    random_image_file = os.path.join(images_dir, random_artist, random_image)

    image = plt.imread(random_image_file)

    axes[i].imshow(image)

    axes[i].set_title("Artist: " + random_artist.replace('_', ' '))

    axes[i].axis('off')



plt.show()
#to train the classificator tried to use ResNet50, an already trained neaural network

## https://www.kaggle.com/suniliitb96/tutorial-keras-transfer-learning-with-resnet50

import tensorflow as tf

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import *

from tensorflow.keras.optimizers import *

from tensorflow.keras.applications import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_input_shape = (224, 224, 3)


#this has to be low so less memory is used

## The kaggle kernel memory was filling and the training stopped everytime, reduced batch size to 5

batch_size = 8

gener = ImageDataGenerator(validation_split=0.2,

                                   rescale=1./255.,

                                   rotation_range=30,

                                   shear_range=5,

                                   zoom_range=0.2,

                                   horizontal_flip=True,

                                   vertical_flip=True,

                                  )



#this creates random images to check the training accuracy

train_generator = gener.flow_from_directory(directory=images_dir,class_mode='categorical',target_size=(224,224),

                                                    batch_size=batch_size,subset="training",shuffle=True,classes=artists_top_name.tolist()

                                                   )



#this creates random images to check the validation accuracy 

valid_generator = gener.flow_from_directory(directory=images_dir, class_mode='categorical', target_size=(224,224), batch_size=batch_size,

                                                    subset="validation", shuffle=True, classes=artists_top_name.tolist()

                                                   )



STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size

STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size



image_dir = "/kaggle/input/best-artworks-of-all-time/images/images/Peter_Paul_Rubens/Peter_Paul_Rubens_141.jpg"

image = plt.imread(image_dir)

image_transf = gener.random_transform(image)

fig, axes = plt.subplots(1, 2, figsize=(15,5))

axes[0].imshow(image)

axes[1].imshow(image_transf)

plt.show()
# Prediction

from keras.preprocessing import *



#function to view the results of some random painting recognition

def testPaintings(model, numberofimg):

    fig, axes = plt.subplots(1, numberofimg, figsize=(20,20))



    for i in range(numberofimg):

        random_artist = random.choice(artists_top_name)

        random_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))

        random_image_file = os.path.join(images_dir, random_artist, random_image)



        # Original image



        test_image = image.load_img(random_image_file, target_size=(train_input_shape))



        # Predict artist

        test_image = image.img_to_array(test_image)

        test_image = test_image//255

        test_image = np.expand_dims(test_image, axis=0)



        prediction = model.predict(test_image)

        prediction_probability = np.amax(prediction)

        prediction_idx = np.argmax(prediction)



        labels = train_generator.class_indices

        labels = dict((v,k) for k,v in labels.items())



        title = "Actual artist = " + str(random_artist) + "\nPredicted artist = " + str(labels[prediction_idx]) + "\nPrediction probability = " +  str(prediction_probability*100)



        # Print image

        axes[i].imshow(plt.imread(random_image_file))

        axes[i].set_title(title)

        axes[i].axis('off')



        plt.show()

    

        labels = train_generator.class_indices

        labels = dict((v,k) for k,v in labels.items())



        title = "Actual artist = " + str(random_artist) + "\nPredicted artist = " + str(labels[prediction_idx]) + "\nPrediction probability = " +  str(prediction_probability*100)



        # Print image

        axes[i].imshow(plt.imread(random_image_file))

    return

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=train_input_shape)

#base_model.summary()
# Add layers at the end

X = base_model.output

X = Flatten()(X)



X = Dense(512, kernel_initializer='he_uniform')(X)

X = BatchNormalization()(X)

X = Activation('relu')(X)

X = Dense(16, kernel_initializer='he_uniform')(X)

X = BatchNormalization()(X)

X = Activation('relu')(X)



n_classes = artists_top.shape[0]



output = Dense(n_classes, activation='softmax')(X)



model = Model(inputs=base_model.input, outputs=output)



model.summary()
for layer in model.layers:

    layer.trainable = True



optimizer = Adam(lr=0.0001)



model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])



#number of training epochs

n_epoch = 10





# Train the model - all layers

# https://keras.io/models/sequential/

# fit_generator trains the model on data generated by a Python generator that runs in parallel

results= model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,epochs=n_epoch,shuffle=True,verbose=1,use_multiprocessing=True,workers=1,)



model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")



#TO LOAD THE MODEL

#json_file = open('model.json', 'r')

#loaded_model_json = json_file.read()

#json_file.close()

#loaded_model = model_from_json(loaded_model_json)

# load weights into new model

#loaded_model.load_weights("model.h5")

#print("Loaded model from disk")

#see accuracy changes



acc = results.history['accuracy']

val_acc = results.history['val_accuracy']

loss = results.history['loss']

val_loss = results.history['val_loss']

epoche = range(len(acc))



fig, axes = plt.subplots(1, 2, figsize=(15,5))



axes[0].plot(epoche, acc, 'r-', label='Training Accuracy')

axes[0].plot(epoche, val_acc, 'b--', label='Validation Accuracy')

axes[0].set_title('Training and Validation Accuracy')

axes[0].legend(loc='best')



axes[1].plot(epoche, loss, 'r-', label='Training Loss')

axes[1].plot(epoche, val_loss, 'b--', label='Validation Loss')

axes[1].set_title('Training and Validation Loss')

axes[1].legend(loc='best')



plt.show()
#check the accuracy on some paintings (of dataset)

testPaintings(model, 5)