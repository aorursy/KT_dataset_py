import tensorflow as tf

from tensorflow.python import keras



print('Tensorflow Version: ', tf.__version__)

print('Keras Version: ', keras.__version__)
from tensorflow.python.keras.applications import VGG16

import os

import numpy as np

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras import models

from tensorflow.python.keras import layers

from tensorflow.python.keras import optimizers



def get_model():

    model = models.Sequential()

    conv_base=VGG16(weights='imagenet',include_top=False,input_shape=(100,100,3))

    model.add(conv_base)

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=1e-4),metrics=['accuracy'])

    return model
image_height = 100

image_width = 100

batch_size = 4

no_of_epochs  = 10
from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt

import os

import cv2

%matplotlib inline



DATADIR = '/kaggle/input/pulmonary-chest-xray-abnormalities/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png'

data=[]

for img in tqdm(os.listdir(DATADIR)):

    try:

        img_array = cv2.imread(os.path.join(DATADIR,img))

        img_array = cv2.resize(img_array, (image_height, image_width))

        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)        

        img_array = img_array.astype(np.float32)/255.

        if img[-5]=='0':

            data.append([img_array, 0])

        else:

            data.append([img_array, 1])

    except Exception as e:   

            pass

print(len(data))
import random



random.shuffle(data)

for sample in data[:10]:

    print(sample[1])
X = []

y = []



for features,label in data:

    X.append(features)

    y.append(label)



X = np.array(X).reshape(-1, image_width, image_height, 3)

print(X.shape)
k=10

num_validation_samples=len(X)//k

validation_scores=[]

for fold in range(k):

    validation_data=X[num_validation_samples*fold:num_validation_samples*(fold+1)]

    validation_labels=y[num_validation_samples*fold:num_validation_samples*(fold+1)]

    if fold==0:

        training_data=X[num_validation_samples*(fold+1):]

        training_labels=y[num_validation_samples*(fold+1):]    

    else:

        training_data=np.append(X[:num_validation_samples*fold], X[num_validation_samples*(fold+1):],axis=0)

        training_labels=np.append(y[:num_validation_samples*fold], y[num_validation_samples*(fold+1):],axis=0)

    model=get_model()

    model.fit(training_data,training_labels,batch_size=batch_size,epochs=no_of_epochs) # 50 epochs per model

    validation_score=model.evaluate(validation_data,validation_labels)

    validation_scores.append(validation_score[1])
print('Average Validation Score: ', np.average(validation_scores))