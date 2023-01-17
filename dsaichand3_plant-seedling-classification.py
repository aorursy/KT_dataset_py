import cv2

import numpy as np

import pandas as pd



import matplotlib.pyplot as plot

%matplotlib inline
import tensorflow

from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

from keras.preprocessing.image import ImageDataGenerator
import os

os.listdir("../input/")
def get_images(directory):

    Images = []

    Labels = []

    for dir_name in os.listdir(directory): 

        for image_file in os.listdir(directory+dir_name):

            image = cv2.imread(directory+dir_name+r'/'+image_file)

            if image is not None:

                image = cv2.resize(image,(300,300),)

                Images.append(image)

                Labels.append(dir_name)

    return Images, Labels
Images, Labels = get_images('../input/train/')
labels = []

mapping = { 'Sugar beet': 0, 'Fat Hen': 1, 'Scentless Mayweed' : 2, 'Charlock' : 3,

           'Small-flowered Cranesbill': 4, 'Maize': 5, 'Shepherds Purse' :6, 'Common wheat': 7,

           'Common Chickweed': 8, 'Cleavers': 9, 'Loose Silky-bent' : 10, 'Black-grass': 11 }

for label in Labels:

    labels.append(mapping[label])

del Labels
Images[0].shape
Images = np.reshape(Images,(-1,300,300,3))

Labels = np.array(labels)
print("Shape of training data: ", Images.shape)

print("Shape of labels data: ", Labels.shape)
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(Images, Labels, test_size=.2, random_state=42, stratify = Labels)
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train,num_classes=12)

y_val = np_utils.to_categorical(y_val,num_classes=12)
train_datagen = ImageDataGenerator(

                                   rotation_range=20,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                  )



validation_datagen = ImageDataGenerator()
del Images

del Labels
train_generator = train_datagen.flow(x_train, y_train, batch_size=32)

validation_generator = validation_datagen.flow(x_val, y_val, batch_size=16)
from tensorflow.keras.applications import VGG16

vgg = VGG16(include_top=

            False, weights='imagenet', input_shape = (300,300,3))
import tensorflow.keras.optimizers as Optimizer

from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAvgPool2D, GlobalMaxPooling2D, Concatenate

from tensorflow.keras.models import Model
vgg.trainable=False

for layer in vgg.layers:

    layer.trainable = False



fc1 = Concatenate(axis=-1)([GlobalAvgPool2D()(vgg.output), GlobalMaxPooling2D()(vgg.output)])

fc1 = Dense(400, activation='relu')(fc1)

fc1_dropout = Dropout(0.3)(fc1)

fc2 = Dense(200, activation='relu')(fc1_dropout)

fc2_dropout = Dropout(0.3)(fc2)

fc2 = Dense(75, activation='relu')(fc1_dropout)

output = Dense(12, activation='softmax')(fc2_dropout)

model = Model(vgg.input, output)



model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('saved_model.hdf5', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, mode='min')

callbacks_list = [checkpoint]

trained = model.fit_generator(train_generator,steps_per_epoch = 25, epochs=200, validation_data = validation_generator,

                              validation_steps=10, 

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
def get_test_images(directory):

    Images = []

    Image_names = []

    for image_file in os.listdir(directory):

        Image_names.append(image_file)

        image = cv2.imread(directory+r'/'+image_file)

        if image is not None:

            image = cv2.resize(image,(300,300),)

            Images.append(image)

    return Images, Image_names
test_images, image_names = get_test_images('../input/test/')

test_images = np.array(test_images)

print(test_images.shape)
vgg = VGG16(include_top=

            False, weights='imagenet', input_shape = (300,300,3))



vgg.trainable=False

for layer in vgg.layers:

    layer.trainable = False



fc1 = Concatenate(axis=-1)([GlobalAvgPool2D()(vgg.output), GlobalMaxPooling2D()(vgg.output)])

fc1 = Dense(400, activation='relu')(fc1)

fc1_dropout = Dropout(0.3)(fc1)

fc2 = Dense(200, activation='relu')(fc1_dropout)

fc2_dropout = Dropout(0.3)(fc2)

fc2 = Dense(75, activation='relu')(fc1_dropout)

output = Dense(12, activation='softmax')(fc2_dropout)

model = Model(vgg.input, output)
model.load_weights('saved_model.hdf5')
tensorflow.keras.models.save_model(

    model,

    'tf_model.hdf5',

    overwrite=True,

    include_optimizer=True

)
from tensorflow.keras.models import load_model

model = load_model('tf_model.hdf5')
predictions = model.predict(test_images)

predictions = np.argmax(predictions, axis = 1)
labelled_predictions = []

mapping = {0: 'Sugar beet',1:'Fat Hen' ,2: 'Scentless Mayweed',3:  'Charlock', 

        4:'Small-flowered Cranesbill', 5:'Maize' ,

        6: 'Shepherds Purse' ,7:'Common wheat' ,8:'Common Chickweed' ,

        9:'Cleavers' ,10:'Loose Silky-bent'  ,11: 'Black-grass'}

for pred in predictions:

    labelled_predictions.append(mapping[pred])
d = []

i=0

for pred in labelled_predictions:

    d.append({'file': image_names[i], 'species': pred})

    i=i+1

output = pd.DataFrame(d)

output.to_csv('submission.csv',index=False)