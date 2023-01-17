from numpy.random import seed
seed(111)
#from tensorflow import set_random_seed
#set_random_seed(90)            #when version tensorflow tf 1.x

import tensorflow as tf
tf.compat.v1.set_random_seed(111)        #when version tensorflow tf 2.0

import pandas as pd
import numpy as np

import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import os
import cv2

import imageio
import skimage
import skimage.io
import skimage.transform

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
%matplotlib inline

Image_Width=224
Image_Height=224
Image_Size=(Image_Width,Image_Height)
Image_Channels=3
filenames=os.listdir("../input/cancer-breast/cancer/")
categories=[]
for f_name in filenames:
    category=f_name.split('.')[0]
    if category=='Benign':
        categories.append("Benign")
    else:
        categories.append("malignant")
df=pd.DataFrame({
    'filename':filenames,
    'category':categories
})
print(df)
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
model = Sequential()
model.add(VGG16(weights="imagenet", 
                include_top=False, input_shape=(224, 224, 3)))
model.add(Flatten())
model.add(Dense(8192, activation="relu"))
model.add(Dense(2048, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="sigmoid"))

model.layers[-7].trainable = False
model.compile(loss='categorical_crossentropy',
  optimizer='rmsprop',metrics=['accuracy'])
model.summary()
model.compile(Adam(lr=0.001), loss='binary_crossentropy', 
              metrics=['accuracy'])
###للتقسيم ل 3 اجزاء     
df["category"] = df["category"].replace({1:'Benign',2:'Malignant'})
#X_train, X_test, y_train, y_test = train_test_split(df["filename"], df["category"], test_size=0.2, random_state=10)
train_df,test_df  = train_test_split(df, test_size=0.2, random_state=10)
print((np.array(train_df)).shape[0])
print((np.array(test_df)).shape[0])

# 0.25 x 0.8 = 0.2
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=10)
train_df ,validate_df = train_test_split(train_df, test_size=0.25, random_state=10)
print((np.array(train_df)).shape[0])
print((np.array(validate_df)).shape[0])
batch_size=128
total_train=train_df.shape[0]
total_validate=validate_df.shape[0]
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    "../input/cancer-breast/cancer/",
                                                    x_col="filename",y_col="category",
                                                    target_size=Image_Size,
                                                 class_mode='categorical',
                                                 batch_size=batch_size)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "../input/cancer-breast/cancer/", 
    x_col="filename",y_col="category",
    target_size=Image_Size,
    class_mode='categorical',
    batch_size=batch_size
)
test_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_generator = test_datagen.flow_from_dataframe(test_df,
                                                 "../input/cancer-breast/cancer/",
                                                    x_col="filename",y_col="category",
                                                 target_size=Image_Size,
                                                 class_mode='categorical',
                                                 batch_size=batch_size)
epochs=20
history = model.fit_generator(train_generator,epochs=epochs, steps_per_epoch=total_train//batch_size, 
                              validation_data=validation_generator,
                              validation_steps=total_validate//batch_size)
model.save("/kaggle/working/mymodel.h5")
results = model.evaluate(test_generator, batch_size=128)
print(results)
import matplotlib.pyplot as plt

plt.plot(history.history['val_loss'])

plt.plot(history.history['loss'])

plt.xlabel('Loss')

plt.ylabel('Iterations')

plt.show()
#ادخال صوره للموديل وتحديد اسمها  
# example of using a pre-trained model as a classifier
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import load_model
# load an image from file
from tensorboard import summary

image = load_img('../input/cancer-breast/cancer/Benign.1.png', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
####image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image=np.expand_dims(image,axis=0)
# prepare the image for the VGG model
image = preprocess_input(image)
# load the model
#model = VGG16() # /content/drive/My Drive/Dataset_BreastCancer/all_mias_scans.h5
model=load_model("./mymodel.h5")
#model.summary()


# predict the probability across all output classes
yhat = model.predict(image)
print(yhat)
res=np.argmax(yhat)
dict1={0:"Benign",1:"maligant"}
the_result=dict1[res]
print(the_result)
