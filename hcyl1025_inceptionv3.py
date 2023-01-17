# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
reminder = "v15:too much overfit.the model learn 10% only!"\
           "Normalization help to stablize the train process and delay overfit"\
            "using new preprocess technic"
random_seed = 8
#-------------------------------------------------------------------
#system
import os
import gc
import warnings
warnings.filterwarnings("ignore")
#-------------------------------------------------------------------
#basic
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#-------------------------------------------------------------------
#keras
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
#-------------------------------------------------------------------
#computer vision
import cv2 as cv
#-------------------------------------------------------------------
#version
print(keras.__version__)
print(pd.__version__)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
print(os.listdir("../input/"))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# Any results you write to the current directory are saved as output.
TrainLab = pd.read_table('../input/dataseta-train-20180813/dataseta_train_20180813/DatasetA_train_20180813/DatasetA_train_20180813/label_list.txt',sep='\t')
lableList = list(TrainLab.label)
print(len(lableList))
#create a data generator object with some image augmentation specs
dataGen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip = True,
    preprocessing_function = preprocess_input,
    validation_split = 0.2)
train_generator  = dataGen.flow_from_directory(
        "../input/alreadysorted/alreadysorted/AlreadySorted/",
        target_size=(299, 299),
        batch_size=64,
        classes = lableList,
        class_mode='categorical',
        seed = 8,
        subset='training')
validation_generator  = dataGen.flow_from_directory(
        "../input/alreadysorted/alreadysorted/AlreadySorted/",
        target_size=(299, 299),
        batch_size=64,
        classes = lableList,
        class_mode='categorical',
        seed = 8,
        subset='validation')
from keras.models import load_model,Model
model = load_model("../input/inceptionv3/ESpretrainedInceptionV3_model.h5")
model.summary()
file_path="ESpretrainedInceptionV3_model.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=15)
callbacks_list = [checkpoint, early] #early
with tf.device('/gpu:0'):
    history = model.fit_generator(generator=train_generator, epochs=10, shuffle=True, validation_data=validation_generator,verbose=1,callbacks=callbacks_list)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()