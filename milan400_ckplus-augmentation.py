import numpy as np

import pickle

import cv2

from os import listdir

from sklearn.preprocessing import LabelBinarizer

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation, Flatten, Dropout, Dense

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing import image

from keras.preprocessing.image import img_to_array

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping, ModelCheckpoint
EPOCHS = 5

BS = 32

default_image_size = tuple((128, 128))

image_size = 0

directory_root = '../input/ckplus-dataset/ck/CK+48'

width=256

height=256

depth=3
def convert_image_to_array(image_dir):

    try:

        image = cv2.imread(image_dir)

        if image is not None :

            image = cv2.resize(image, default_image_size)   

            return img_to_array(image)

        else :

            return np.array([])

    except Exception as e:

        print(f"Error : {e}")

        return None
image_list,label_list = [],[]

try:

    print("loading images...")

    root_dir = listdir(directory_root)

    for emotion_folder in root_dir:

        print(f"[INFO] Processing {emotion_folder}")

        images_list = listdir(f"{directory_root}/{emotion_folder}")

        

        #for emotion_image in emotion_folder_list:

        for images in images_list:

            image = f"{directory_root}/{emotion_folder}/{images}"

            image_list.append(convert_image_to_array(image))

            label_list.append(emotion_folder)

except Exception as e:

    print(f"Error : {e}")
import pandas as pd

a = pd.DataFrame(label_list)

idx = pd.Index(a)

count = idx.value_counts()

print(count)
image_size = len(image_list)

print(image_size)
np_image_list = np.array(image_list,dtype=np.float32)/255.0
label_binarizer = LabelBinarizer()

image_labels = label_binarizer.fit_transform(label_list)

pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))

n_classes = len(label_binarizer.classes_)

x_train,x_test,y_train,y_test = train_test_split(np_image_list,image_labels,test_size=0.01,random_state=42)
x_train[0].shape
aug = ImageDataGenerator(

    rotation_range=25, width_shift_range=0.1,

    height_shift_range=0.1, shear_range=0.2, 

    zoom_range=0.2,horizontal_flip=True, 

    fill_mode="nearest")
from keras.applications import VGG19

from keras.optimizers import SGD,RMSprop,adam,Adadelta

#Load the VGG model

vgg_conv = VGG19(weights=None, include_top=False, input_shape=(128, 128,3))
def vgg_custom():

    model = Sequential()

    #add vgg conv model

    model.add(vgg_conv)

    

    #add new layers

    model.add(Flatten())

    model.add(Dense(7,  kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer=Adadelta())

    

    return model
model = vgg_custom()

model.summary()
from keras import callbacks

filename='model_train_new.csv'

filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"



csv_log=callbacks.CSVLogger(filename, separator=',', append=False)

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [csv_log,checkpoint]

callbacks_list = [csv_log]
history = model.fit_generator(

    aug.flow(x_train, y_train, batch_size=BS),

    validation_data=(x_test, y_test),

    steps_per_epoch=len(x_train) // BS,

    epochs=EPOCHS, verbose=1,

    callbacks = callbacks_list #early stopping

    )
no_images = 0

no_sadness = 0

no_anger = 0

no_disgust = 0

no_happy = 0

no_fear = 0

no_surprise = 0

no_contempt = 0

for e in range(2):

    print('Epoch', e)

    batches = 0

    for x_batch, y_batch in aug.flow(x_train, y_train, batch_size=32):

        model.fit(x_batch, y_batch)

        batches += 1

        no_images +=len(x_batch)

        y_batch_real = label_binarizer.inverse_transform(y_batch)

        for label in y_batch_real:

            if(label== 'sadness'):

                no_sadness+=1

            elif(label=='anger'):

                no_anger+=1

            elif(label=='disgust'):

                no_disgust+=1

            elif(label=='happy'):

                no_happy+=1

            elif(label=='fear'):

                no_fear+=1

            elif(label=='surprise'):

                no_surprise+=1

            elif(label=='contempt'):

                no_contempt+=1

        if batches >= len(x_train) / 32:

            # we need to break the loop by hand because

            # the generator loops indefinitely

            break
print(len(x_train))
print(no_images)
print(no_anger)
print(no_disgust)
print(no_fear)
print(no_happy)
print(no_sadness)
print(no_surprise)
print(no_contempt)