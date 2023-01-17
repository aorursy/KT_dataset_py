import numpy as np 

import pandas as pd

import matplotlib.pyplot as plot

import cv2

from random import randint

from sklearn.utils import shuffle



import os

print(os.listdir("../input"))
import tensorflow.keras.models as Models

import tensorflow.keras.layers as Layers

import tensorflow.keras.activations as Activations

import tensorflow.keras.optimizers as Optimizer



from keras.preprocessing.image import ImageDataGenerator
def get_images(directory):

    Images = []

    Labels = []

    for dir_name in os.listdir(directory):

        for image_file in os.listdir(directory+dir_name):

            image = cv2.imread(directory+dir_name+r'/'+image_file)

            image = cv2.resize(image,(150,150),)

            Images.append(image)

            Labels.append(dir_name)

    return shuffle(Images,Labels,random_state=817328462)
Images, Labels = get_images('../input/seg_train/seg_train/')
mapping = {"buildings":0, "glacier":2, "forest": 1, "mountain": 3, "sea": 4, "street": 5}

labels = []

for label in Labels:

    labels.append(mapping[label])

del Labels
Images = np.array(Images)

Labels = np.array(labels)

print("Shape of Images: ",Images.shape)

print("Shape of Labels: ",Labels.shape)
from keras.utils import np_utils

Labels = np_utils.to_categorical(Labels,num_classes=6)
print("Shape of Images: ",Images.shape)

print("Shape of Labels: ",Labels.shape)
train_Images = Images[0:11000]

train_Labels = Labels[0:11000]

valid_Images = Images[11000:]

valid_Labels = Labels[11000:]
train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_Images, train_Labels, batch_size=30)

validation_generator = validation_datagen.flow(valid_Images, valid_Labels, batch_size=20)
f,ax = plot.subplots(5,5) 

f.subplots_adjust(0,0,3,3)

for i in range(0,5,1):

    for j in range(0,5,1):

        rnd_number = randint(0,len(train_Images))

        ax[i,j].imshow(train_Images[rnd_number])

        ax[i,j].set_title(train_Labels[rnd_number])

        ax[i,j].axis('off')
from tensorflow.keras.applications import VGG16

vgg = VGG16(include_top=False, weights='imagenet', input_shape = (150,150,3))
from tensorflow.keras.layers import Flatten, Dense, Dropout

from tensorflow.keras.models import Model



vgg.trainable=False

for layer in vgg.layers[:-4]:

    layer.trainable = False



vgg_output = vgg.output



fc1 = Flatten()(vgg_output)

fc1 = Dense(512, activation='relu')(fc1)

fc1_dropout = Dropout(0.3)(fc1)

fc2 = Dense(512, activation='relu')(fc1_dropout)

fc2_dropout = Dropout(0.3)(fc2)

output = Dense(6, activation='softmax')(fc2_dropout)

model = Model(vgg.input, output)



model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
layers = [(layer, layer.name,layer.trainable) for layer in model.layers]

pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('saved_model_3.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]

trained = model.fit_generator(train_generator,steps_per_epoch=100, epochs=10, validation_data = validation_generator,validation_steps=50, 

                              verbose=1, callbacks = callbacks_list)
plot.plot(trained.history['acc'])

plot.plot(trained.history['val_acc'])

plot.title('Model accuracy')

plot.ylabel('Accuracy')

plot.xlabel('Epoch')

plot.legend(['Train', 'Test'], loc='upper left')

plot.show()



plot.plot(trained.history['loss'])

plot.plot(trained.history['val_loss'])

plot.title('Model loss')

plot.ylabel('Loss')

plot.xlabel('Epoch')

plot.legend(['Train', 'Test'], loc='upper left')

plot.show()
test_images,test_labels = get_images('../input/seg_test/seg_test/')

test_images = np.array(test_images)

test_labels = np.array(test_labels)
mapping = {"buildings":0, "glacier":2, "forest": 1, "mountain": 3, "sea": 4, "street": 5}

pred_labels = []

for label in test_labels:

    pred_labels.append(mapping[label])
pred_labels = np_utils.to_categorical(pred_labels)
from tensorflow.keras.models import load_model

model = load_model('saved_model_3.hdf5')

model.evaluate(test_images,pred_labels, verbose=1)
def get_pred_images(directory):

    Images = []

    Image_names = []

    for image_file in os.listdir(directory):

        Image_names.append(image_file)

        image = cv2.imread(directory+r'/'+image_file)

        image = cv2.resize(image,(150,150))

        Images.append(image)

    return Images, Image_names
pred_images, Image_names = get_pred_images('../input/seg_pred/seg_pred/')

#model.predict()
pred_images = np.asarray(pred_images)

pred_images = pred_images/255

print(pred_images.shape)
predictions = model.predict(pred_images)

predictions = np.argmax(predictions, axis = 1)

d = []

i=0

for pred in predictions:

    d.append({'image_name': Image_names[i], 'label': pred})

    i=i+1

output = pd.DataFrame(d)

output.to_csv('submission.csv',index=False)