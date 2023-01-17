from os import listdir

import numpy as np

import cv2

import pickle

import matplotlib.pyplot as plt

from random import shuffle 

from tqdm import tqdm 

import tensorflow as tf

import matplotlib.pyplot as plt

from keras import layers

from keras.layers import Input

from keras.layers import Add

from keras.layers import Dense

from keras.layers import Activation

from keras.layers import ZeroPadding2D

from keras.layers import BatchNormalization

from keras.layers import Flatten

from keras.layers import Conv2D

from keras.layers import AveragePooling2D

from keras.layers import MaxPooling2D

from keras.models import Model, load_model

from keras.layers import Dropout

from keras.callbacks import ModelCheckpoint

from keras.losses import binary_crossentropy

from keras.losses import sparse_categorical_crossentropy

from keras.losses import categorical_crossentropy

from keras.models import Model, load_model

from keras.models import Sequential

from keras.layers import ZeroPadding2D

import keras.backend as K

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing import image

from keras.preprocessing.image import img_to_array
EPOCHS = 20

LR = 1e-3

BS = 32

default_image_size = tuple((224, 224))

image_size = 0

directory_root = '../input/plantdisease' 

width=224

height=224

depth=3
# converting images into an array 



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
# Creating an empty image and label list 



image_list, label_list = [], []



# Cleaning data for tomato leaves and fetching images from the required directories

try:

    print("[INFO] Loading images ...")

    root_dir = listdir(directory_root) 

    for directory in root_dir :

        if directory == ".DS_Store" :

            root_dir.remove(directory)



    for plant_folder in root_dir :

        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")

        

        for disease_folder in plant_disease_folder_list :

            if disease_folder == ".DS_Store" :

                plant_disease_folder_list.remove(disease_folder)



        for plant_disease_folder in plant_disease_folder_list:

            print(f"[INFO] Processing {plant_disease_folder} ...")

            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")

                

            for single_plant_disease_image in plant_disease_image_list :

               if single_plant_disease_image == ".DS_Store" :

                    plant_disease_image_list.remove(single_plant_disease_image)



            for image in plant_disease_image_list[:300]:

                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"

                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:

                    image_list.append(convert_image_to_array(image_directory))

                    label_list.append(plant_disease_folder)

    print("[INFO] Image loading completed")  

except Exception as e:

    print(f"Error : {e}")
image_size = len(image_list)

print(image_size)
label_binarizer = LabelBinarizer()

image_labels = label_binarizer.fit_transform(label_list)

pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))

n_classes = len(label_binarizer.classes_)
print(label_binarizer.classes_)
np_image_list = np.array(image_list, dtype=np.float16) / 225.0
print("[INFO] Spliting data for training and testing")

x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42)
print(x_train.shape)
aug = ImageDataGenerator(

    rotation_range=25, width_shift_range=0.1,

    height_shift_range=0.1, shear_range=0.2, 

    zoom_range=0.2,horizontal_flip=True, 

    fill_mode="nearest")
model = Sequential()

inputShape = (height, width, depth)

chanDim = -1

if K.image_data_format() == "channels_first":

    inputShape = (depth, height, width)

    chanDim = 1

model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024))

model.add(Activation("relu"))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(n_classes))

model.add(Activation("softmax"))



# Compile the model

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
model.summary()
history = model.fit_generator(

    aug.flow(x_train, y_train, batch_size=BS),

    validation_data=(x_test, y_test),

    steps_per_epoch=len(x_train) // BS,

    epochs=EPOCHS, verbose=1

    )
accuracy = history.history['categorical_accuracy']

val_accuracy = history.history['val_categorical_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(accuracy) + 1)



#Train and validation accuracy

plt.plot(epochs, accuracy, 'b', label='Training accurarcy')

plt.plot(epochs, val_accuracy, 'r', label='Validation accurarcy')

plt.title('Training and Validation accurarcy')

plt.legend()

plt.figure()
#Train and validation loss

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and Validation loss')

plt.legend()

plt.show()
from keras.models import model_from_json

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")
from __future__ import division, print_function

# coding=utf-8

import sys

import os

import glob

import re

import numpy as np



# Keras

from keras.applications.imagenet_utils import preprocess_input, decode_predictions

from keras.models import load_model

from keras.preprocessing import image



# Flask utils

from flask import Flask, redirect, url_for, request, render_template

from werkzeug.utils import secure_filename

from gevent.pywsgi import WSGIServer



# Define a flask app

app = Flask(__name__)



# Model saved with Keras model.save()

MODEL_PATH = './model.h5'



# Load your trained model

model = load_model(MODEL_PATH)

model._make_predict_function()          # Necessary

print('Model loaded. Start serving...')



print('Model loaded. Check http://127.0.0.1:5000/')





def model_predict(img_path, model):

    img = image.load_img(img_path, target_size=(224, 224))



    # Preprocessing the image

    x = image.img_to_array(img)

    # x = np.true_divide(x, 255)

    x = np.expand_dims(x, axis=0)



    # Be careful how your trained model deals with the input

    # otherwise, it won't make correct prediction!

    x = preprocess_input(x, mode='caffe')



    preds = model.predict(x)

    return preds





@app.route('/', methods=['GET'])

def index():

    # Main page

    return render_template('../input/html-index')





@app.route('/predict', methods=['GET', 'POST'])

def upload():

    if request.method == 'POST':

        # Get the file from post request

        f = request.files['file']



        # Save the file to ./uploads

        basepath = os.path.dirname(__file__)

        file_path = os.path.join(

            basepath, 'uploads', secure_filename(f.filename))

        f.save(file_path)



        # Make prediction

        preds = model_predict(file_path, model)



        # Process your result for human

        # pred_class = preds.argmax(axis=-1)            # Simple argmax

        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        result = str(pred_class[0][0][1])               # Convert to string

        return result

    return None





if __name__ == '__main__':

    app.run(debug=True)
