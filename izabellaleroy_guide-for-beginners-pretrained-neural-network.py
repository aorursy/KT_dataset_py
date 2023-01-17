import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import cv2

import tensorflow as tf 

import pandas as pd

import numpy as np

from tensorflow.keras.models import  Model

from tensorflow.keras.layers import Dropout, Dense

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import load_model
images = [[],[],[]]

index = ['001', '002', '003']



for i in range(3):

    images[i].append(mpimg.imread('../input/lego-minifigures-classification/star-wars/0001/' + index[i] + '.jpg'))

    images[i].append(mpimg.imread('../input/lego-minifigures-classification/marvel/0001/' + index[i] + '.jpg'))

    images[i].append(mpimg.imread('../input/lego-minifigures-classification/harry-potter/0001/' + index[i] + '.jpg'))



fig, axs = plt.subplots(3,3, figsize=(18,18))



for i in range(3):

    for j in range(3):

        axs[i, j].imshow(images[i][j])
data = pd.read_csv('../input/lego-minifigures-classification/index.csv')

data
train_set = data[data["train-valid"] == 'train']

validation_set = data[data["train-valid"] == 'valid']
# take pretrained DenseNet

base_model = tf.keras.applications.DenseNet121()

# Create new Dropout layer and sent there penultimate output from DenseNet 

my_layer = Dropout(0.5)(base_model.layers[-2].output)

# Count the nuber of unique classes in dataset

number_of_classes = len(data['class_id'].unique())

# Create new Dense layer and sent there output from Dropout layer

# Note that in this Dense layer is "number_of_classes" neuron because we have "number_of_classes" classes 

my_outputs = Dense(number_of_classes, activation="softmax")(my_layer)

model = Model(base_model.input, my_outputs)
model.compile(loss='sparse_categorical_crossentropy',

              optimizer=Adam(0.0001),

              metrics=['accuracy'])


X_train = np.zeros((train_set.shape[0], 512, 512, 3))



for i in range(train_set.shape[0]):

    image = cv2.imread('../input/lego-minifigures-classification/' + train_set["path"].values[i])

    image = cv2.resize(image, dsize=(512,512)) # resize in case if image was not 512x512 pixels

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    X_train[i] = image/255



Y_train = np.array(train_set["class_id"])-1



X_valid = np.zeros((validation_set.shape[0], 512, 512, 3))



for i in range(validation_set.shape[0]):

    image = cv2.imread('../input/lego-minifigures-classification/' + validation_set["path"].values[i])

    image = cv2.resize(image, dsize=(512,512))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    X_valid[i] = image/255



Y_valid = np.array(validation_set["class_id"])-1
Y_train
checkpoint = ModelCheckpoint(filepath='model.h5', monitor="val_accuracy", save_best_only=True, verbose=1)
model.fit(

    X_train, 

    Y_train, 

    epochs=50, 

    validation_data=(X_valid, Y_valid), 

    shuffle=True, 

    batch_size=4, 

    callbacks=checkpoint

)
model = load_model("model.h5")
image = cv2.imread('../input/lego-minifigures-classification/harry-potter/0002/009.jpg') # read the image 

image = cv2.resize(image, dsize=(512,512)) # resize in case if image was not 512x512 pixels

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255 # transform picture from BRG to the RGB format



plt.imshow(image) # print image 



image = np.reshape(image, (1, 512, 512, 3)) # resize to the needed for model shape - 1 picture 512 height 512 width 3 chanel(RGB)



ans = model.predict(image).argmax() # find index of max element

ans = ans+1 # don't forget to add 1 :) 

metadata = pd.read_csv('../input/lego-minifigures-classification/metadata.csv') # download meta data, there are store real 

                                                                                #names of minifigures



minifigure = metadata["minifigure_name"][metadata["class_id"] == ans].iloc[0] # find the name that matches the predicted class

print(f"Class:\t{ans}\tMinifigure:\t{minifigure}")