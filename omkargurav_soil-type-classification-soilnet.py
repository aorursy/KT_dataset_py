#!pip install tensorflow==2.3.0
!pip install imutils
!pip install python-telegram-bot #For training updates on Telegram

import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dropout, MaxPooling2D, AveragePooling2D, Dense, Flatten, Input, Conv2D, add, Activation
from tensorflow.keras.layers import (Dense, Dropout, Activation, Flatten, Reshape, Layer,
                          BatchNormalization, LocallyConnected2D,
                          ZeroPadding2D, Conv2D, MaxPooling2D, Conv2DTranspose,AveragePooling2D,
                          GaussianNoise, UpSampling2D, Input)

from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential , Model , load_model
from tensorflow.keras.preprocessing.image import load_img , img_to_array , ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from PIL import Image
import matplotlib.pyplot as plt

import cv2
from imutils import paths
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")
print("Tensorflow version: ",tf.__version__)
import requests
import tensorflow as tf

import tensorflow.keras.utils as np_utils

access_token = '' #Access token here

class botCallback(tf.keras.callbacks.Callback):
    def __init__(self,access_token):
        self.access_token = access_token
        self.ping_url = 'https://api.telegram.org/bot'+str(self.access_token)+'/getUpdates'
        self.response = requests.get(self.ping_url).json()
        #print(self.response)
        self.chat_id = self.response['result'][0]['message']['chat']['id']
        #self.chat_id = self.response['result']

    def send_message(self,message):
        #print('sending message')
        self.ping_url = 'https://api.telegram.org/bot'+str(self.access_token)+'/sendMessage?'+\
                        'chat_id='+str(self.chat_id)+\
                        '&parse_mode=Markdown'+\
                        '&text='+message
        self.response = requests.get(self.ping_url)
    
    def send_photo(self,filepath):
        imagefile= open(filepath,"rb")
        file_dict = {'photo':imagefile}
        self.ping_url = 'https://api.telegram.org/bot'+str(self.access_token)+'/sendPhoto?chat_id='+str(self.chat_id)
        self.response = requests.post(self.ping_url, files = file_dict)
        imagefile.close()

    def on_train_batch_begin(self, batch, logs=None):
        pass
    
    def on_train_batch_end(self, batch, logs=None):
        message = ' Iteration/Batch {}\n Training Accuracy : {:7.2f}\n Training Loss : {:7.2f}\n'.format(batch,logs['accuracy'],logs['loss'])
        #print(logs)
        try:
            message += ' Validation Accuracy : {:7.2f}\n Validation Loss : {:7.2f}\n'.format(logs['val_accuracy'],logs['val_loss'])
            self.send_message(message)
        except:
            pass

    def on_test_batch_begin(self, batch, logs=None):
        pass
    
    def on_test_batch_end(self, batch, logs=None):
        message = ' Iteration/Batch {}\n Training Accuracy : {:7.2f}\n Training Loss : {:7.2f}\n'.format(batch,logs['accuracy'],logs['loss'])
        try:
            message += ' Validation Accuracy : {:7.2f}\n Validation Loss : {:7.2f}\n'.format(logs['val_accuracy'],logs['val_loss'])
            self.send_message(message)
        except:
            pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):

        message = ' Epoch {}\n Training Accuracy : {:7.2f}\n Training Loss : {:7.2f}\n'.format(epoch,logs['accuracy'],logs['loss'])
        try:
            message += ' Validation Accuracy : {:7.2f}\n Validation Loss : {:7.2f}\n'.format(logs['val_accuracy'],logs['val_loss'])
            self.send_message(message)        
        except:
            pass

class Plotter(botCallback):
 def __init__(self,access_token):
    
     super().__init__(access_token)
 def on_train_begin(self,logs=None):
     self.batch = 0
     self.epoch = []
     self.train_loss = []
     self.val_loss = []
     self.train_acc = []
     self.val_acc = []
     self.fig = plt.figure(figsize=(200,100))
     self.logs = []

 def on_epoch_end(self, epoch, logs=None):
     self.logs.append(logs)
     self.epoch.append(epoch)
     self.train_loss.append(logs['loss'])
     self.val_loss.append(logs['val_loss'])
     self.train_acc.append(logs['accuracy'])
     self.val_acc.append(logs['val_accuracy'])
     f,(ax1,ax2) = plt.subplots(1,2,sharex=True)
     #clear_output(wait=True)
     ax1.plot(self.epoch, self.train_loss, label='Training Loss')
     ax1.plot(self.epoch, self.val_loss, label='Validation Loss')
     ax1.legend()
     ax2.plot(self.epoch, self.train_acc, label='Training Accuracy')
     ax2.plot(self.epoch, self.val_acc, label='Validation Accuracy')
     ax2.legend()
     plt.savefig('Accuracy and Loss plot.jpg')
     self.send_photo('Accuracy and Loss plot.jpg')

train_dir = '/kaggle/input/soil-classification-image-data/Soil_Dataset/Train'
test_dir = '/kaggle/input/soil-classification-image-data/Soil_Dataset/Test'

image_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255,
                            rotation_range=45,
                            zoom_range=0.40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.15,
                            horizontal_flip=True,
                            vertical_flip= True,
                            fill_mode="nearest")

train_data = train_datagen.flow_from_directory(train_dir,
                                              target_size=(150,150),
                                              batch_size=32,
                                              class_mode="categorical")

test_datagen = ImageDataGenerator(rescale = 1./255)

test_data = test_datagen.flow_from_directory(test_dir,
                                            target_size=(150,150),
                                            batch_size=32,
                                            class_mode="categorical")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                              validation_split = 0.2,
                                                              subset = "training",
                                                              seed = 42,
                                                              image_size = (150,150),
                                                              batch_size = 40)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                             validation_split = 0.2,
                                                             subset = "validation",
                                                             seed = 42,
                                                             image_size = (150,150),
                                                             batch_size = 40)
## Configuring dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
training_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
testing_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = Sequential(name="SoilNet")
model.add(Conv2D(64,(3,3),activation = "relu",padding ="same",kernel_initializer="he_normal", input_shape=(150,150,3)))
#model.add(tf.keras.layers.LeakyReLU())
#model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation = "relu",padding ="same",kernel_initializer="he_normal"))
#model.add(tf.keras.layers.LeakyReLU())
model.add(BatchNormalization())

model.add(AveragePooling2D(pool_size = (2,2), strides=2))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),activation = "relu",padding ="same",kernel_initializer="he_normal"))
#model.add(tf.keras.layers.LeakyReLU())
#model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation = "relu",padding ="same",kernel_initializer="he_normal"))
#model.add(tf.keras.layers.LeakyReLU())
model.add(BatchNormalization())

model.add(AveragePooling2D(pool_size = (2,2), strides=2))
model.add(Dropout(0.5))
model.add(BatchNormalization())

#lk = tf.keras.layers.LeakyReLU()
model.add(Conv2D(256,(3,3),activation = "relu", padding ="same",kernel_initializer="he_normal"))
#model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),activation = "relu",padding ="same",kernel_initializer="he_normal"))
#model.add(tf.keras.layers.LeakyReLU())
model.add(BatchNormalization())

model.add(AveragePooling2D(pool_size = (2,2), strides=2))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.7))
model.add(Dense(4,activation="softmax"))

opt = RMSprop(learning_rate = 0.0001, rho = 0.99, epsilon = 1e-08, decay = 0.0)
model.compile(optimizer=opt,loss="categorical_crossentropy",metrics=["accuracy"])

reduction_lr = ReduceLROnPlateau(monitor = "val_accuracy",patience = 2 ,verbose = 1, factor = 0.3, min_lr = 0.0000001)
reduction_lr1 = ReduceLROnPlateau(monitor = "val_loss",patience = 2 ,verbose = 1, factor = 0.3, min_lr = 0.0000001)
#bot_callback = botCallback(access_token)
#plotter = Plotter(access_token)
#callback_list = [bot_callback,plotter] callbacks=callback_list

start = time.time()

history = model.fit_generator(train_data,
                    validation_data = test_data,
                    epochs=20,
                    callbacks = [reduction_lr,reduction_lr1])
end = time.time()
print("Total train time: ",(end-start)/60," mins")

"""
model = tf.keras.models.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(128, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(2, activation= 'softmax')
])
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
"""
"""
#=================================================================
chanDim = 1
model = Sequential(name="SoilNet")
model.add(Conv2D(32, (3, 3), padding="same",input_shape=(150,150,3)))
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (4, 4), padding="same"))
model.add(Activation("relu"))

model.add(Conv2D(64, (4, 4), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (4, 4), padding="same"))
model.add(Activation("relu"))

model.add(Conv2D(128, (4, 4), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation("softmax"))


model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
reduction_lr = ReduceLROnPlateau(monitor = "val_accuracy",patience = 2 ,verbose = 1, factor = 0.2, min_lr = 0.00001)
callback_list = [reduction_lr]
model.summary()
plot_model(model,show_shapes=True)
"""

"""
history = model.fit(train_ds,
                    validation_data = test_ds,
                    epochs=5)
"""
def plot_graph(history,string):
    plt.figure(figsize=(12,8))
    plt.plot(history.history[string],label=str(string))
    plt.plot(history.history["val_"+str(string)],label="val_"+str(string))
    plt.xlabel("Epochs")
    plt.ylabel(str(string))
    plt.legend()
    plt.show()
plot_graph(history,"accuracy")
plot_graph(history,"loss")
model.save("SoilNet.h5")
from IPython.display import FileLink
FileLink('SoilNet.h5')



