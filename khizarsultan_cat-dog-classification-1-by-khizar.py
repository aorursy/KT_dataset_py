from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D, BatchNormalization,Dropout

from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import optimizers



# from google.colab import files

import zipfile

import tensorflow as tf

import os



%matplotlib inline

import pandas as pd



from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt

import numpy as np



# predicting the images

from PIL import Image

import requests

import cv2

from io import BytesIO

from warnings import filterwarnings

filterwarnings('ignore')
# _URL = 'https://github.com/KhizarSultan/sleep_data_cnn_classification/raw/master/sleep.zip'

# drive url = https://drive.google.com/open?id=1ufX5DhyDmFIa4FgPEdxonACHTRvJHrez

id = "1ufX5DhyDmFIa4FgPEdxonACHTRvJHrez"
from pydrive.auth import GoogleAuth

from pydrive.drive import GoogleDrive

from google.colab import auth

from oauth2client.client import GoogleCredentials
auth.authenticate_user()

gauth = GoogleAuth()

gauth.credentials = GoogleCredentials.get_application_default()

drive = GoogleDrive(gauth)
downloaded = drive.CreateFile({'id':id})   # replace the id with id of file you want to access

downloaded.GetContentFile('cat-dog.zip')
with zipfile.ZipFile("cat-dog.zip","r") as zip_ref:

    zip_ref.extractall("/root/.keras/datasets/") 
training_dir = os.path.join("/root/.keras/datasets/cat-dog","train")

validation_dir = os.path.join("/root/.keras/datasets/cat-dog","validate")

test_dir = os.path.join("/root/.keras/datasets/cat-dog","test")
cat_path = os.path.join(training_dir,"cat")

cat_train_images = len(os.listdir(cat_path))



dog_path = os.path.join(training_dir,"dog")

dog_train_images = len(os.listdir(dog_path))
print(f"Cat Training images (80%) : {cat_train_images}")

print(f"Dog Training images (80%): {dog_train_images}")
cat_path = os.path.join(validation_dir,"cat")

cat_test_images = len(os.listdir(cat_path))



dog_path = os.path.join(validation_dir,"dog")

dog_test_images = len(os.listdir(dog_path))

print(f"Cat Validation images (20%): {cat_test_images}")

print(f"Dog Validation images (20%): {dog_test_images}")

# print(f"Rem Validation images (20%): {rem_test_images}")
mix_path = os.path.join(test_dir,"test1")

mix_test_images = len(os.listdir(mix_path))
print(f"Mix images (20%): {mix_test_images}")
total_train = cat_train_images+dog_train_images

total_test = cat_test_images+dog_test_images



print(f"Total Dog Images {dog_test_images+dog_train_images}")

print(f"Total Cat Images {cat_test_images+cat_train_images}")



print(f"Total Training Images {total_train}")

print(f"Total Validation Images {total_test}")
batch_size = 32

IMG_WIDTH,IMG_HEIGHT = 64,64

epoches = 200
classifier = Sequential()



classifier.add(Conv2D(32,(3,3),input_shape=(IMG_WIDTH,IMG_HEIGHT,3),activation = 'relu'))

classifier.add(BatchNormalization()),

classifier.add(MaxPooling2D(pool_size=(2,2),strides=2)) #if stride not given it equal to pool filter size



classifier.add(Conv2D(64,(3,3),activation = 'relu'))

classifier.add(BatchNormalization()),

classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))



classifier.add(Conv2D(128,(3,3),activation = 'relu'))

classifier.add(BatchNormalization()),

classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))



classifier.add(Flatten())

classifier.add(Dropout(0.25))



classifier.add(Dense(units=512,activation='relu'))

classifier.add(BatchNormalization()),

classifier.add(Dropout(0.5))



classifier.add(Dense(units=1,activation='sigmoid'))
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

classifier.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])

#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))


train_datagen = ImageDataGenerator(rescale=1./255,

                                   shear_range=0.1,

                                   zoom_range=0.1,

                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#Training Set

train_set = train_datagen.flow_from_directory(training_dir,

                                             target_size=(IMG_WIDTH,IMG_HEIGHT),

                                             batch_size=batch_size,

                                             class_mode='binary')

#Validation Set

test_set = test_datagen.flow_from_directory(validation_dir,

                                           target_size=(IMG_WIDTH,IMG_HEIGHT),

                                           batch_size = batch_size,

                                           class_mode='binary',

                                           shuffle=False)

#Test Set /no output available

test_set1 = test_datagen.flow_from_directory(test_dir,

                                            target_size=(IMG_WIDTH,IMG_HEIGHT),

                                            batch_size=batch_size,

                                            shuffle=False)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # to avoid overfitting

# Early stopping is basically stopping the training once your loss starts to increase (or in other words validation accuracy starts to decrease

# Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. 

# This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=5, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]
history = None
history = classifier.fit_generator(train_set,

                        steps_per_epoch=800, 

                        epochs = 180,

                        validation_data = test_set,

                        validation_steps = 20,

                        callbacks = callbacks

                        );
classifier.save("cat_dog_96.h5")
files.download("cat_dog_96.h5")
acc = history.history['acc']

val_acc = history.history['val_acc']



loss = history.history['loss']

val_loss = history.history['val_loss']



# epochs_range = range()



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot( loss, label='Training Loss')

plt.plot( val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()