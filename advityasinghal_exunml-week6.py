from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array,ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense ,Conv2D,MaxPooling2D,Flatten,Dropout,GlobalAveragePooling2D
from keras.optimizers import SGD
import cv2 
import os
import random
import shutil
import pprint
import tensorflow as tf 
import matplotlib.pyplot as pyplot
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.backend import softmax
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import preprocess_input


model=tf.keras.models.load_model(
    "../input/bestmodel/best.h5"
)
datagen = ImageDataGenerator(rescale=1.0/255.0,preprocessing_function= preprocess_input)
ts2=datagen.flow_from_directory('../input/intel-image-classification/seg_test/seg_test',
		class_mode='categorical', batch_size=64, target_size=(224, 224),subset="training")


data=model.evaluate(ts2)  
data=model.evaluate(ts2)  
model_aug=tf.keras.models.load_model(
    "../input/meowwoodfd/aug_corrected.h5"
)
datagen2 = ImageDataGenerator(featurewise_center=True)
datagen2.mean = [123.68, 116.779, 103.939]
ts=datagen2.flow_from_directory('../input/intel-image-classification/seg_test/seg_test',
		class_mode='categorical', batch_size=64, target_size=(224, 224),subset="training")
data_aug=model_aug.evaluate(ts)  
model3=tf.keras.models.load_model(
    "../input/meowwoodfd/vgg_good.h5"
)
data_vgg=model3.evaluate(ts2)  