# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import libraries

from glob import glob 
import cv2 # image processing
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # data visualization


import tensorflow as tf
import keras.preprocessing.image

import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.naive_bayes

import sklearn.ensemble
import os;

import cv2 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm  
from  tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import  Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator,image,img_to_array,load_img
import seaborn as sns
from PIL import Image
import os
print(os.listdir("../input/chest-xray-pneumonia/chest_xray"))

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import RepeatVector, Input, Reshape
import numpy as np;
import cv2;
import urllib;



#creating directory for training images
tr_normal = os.listdir('../input/chest-xray-pneumonia/chest_xray/train/NORMAL/') 
tr_infected = os.listdir('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/')
train_path='../input/chest-xray-pneumonia/chest_xray/'+'train/'

#for validation images
v_normal = os.listdir('../input/chest-xray-pneumonia/chest_xray/val/NORMAL/') 
v_infected = os.listdir('../input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/')
val_path= '../input/chest-xray-pneumonia/chest_xray/'+'val/'
#for test images
ts_normal = os.listdir('../input/chest-xray-pneumonia/chest_xray/test/NORMAL/') 
ts_infected = os.listdir('../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/')
test_path= '../input/chest-xray-pneumonia/chest_xray/'+'test/'

# Data Augmentation
datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=20,
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=20,
        class_mode='binary')
data = []
labels = []

for i in tr_infected:
    try:
    
        image = cv2.imread("../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/"+i)
        image_array = Image.fromarray(image , 'RGB')
        resize_img = image_array.resize((224 , 224))
        
        blur = cv2.blur(np.array(resize_img) ,(10,10))
        data.append(np.array(resize_img))
        
        data.append(np.array(blur))
        labels.append(1)
        
        labels.append(1)
        
    except AttributeError:
        print('')
    
for u in tr_normal:
    try:
        
        image = cv2.imread("../input/chest-xray-pneumonia/chest_xray/train/NORMAL/"+u)
        image_array = Image.fromarray(image , 'RGB')
        resize_img = image_array.resize((224 , 224))
        
        data.append(np.array(resize_img))
        
        labels.append(0)
        
        
    except AttributeError:
        print('')
x_ray = np.array(data)
labels = np.array(labels)

np.save('x_ray' , x_ray)
np.save('Labels' , labels)
print('x_ray : {} | labels : {}'.format(x_ray.shape , labels.shape))
plt.figure(1 , figsize = (15 , 9))
n = 0 
for i in range(49):
    n += 1 
    r = np.random.randint(0 , x_ray.shape[0] , 1)
    plt.subplot(7 , 7 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    plt.imshow(x_ray[r[0]])
    plt.title('{} : {}'.format('Pneumonia' if labels[r[0]] == 1 else 'Normal' ,
                               labels[r[0]]) )
    plt.xticks([]) , plt.yticks([])
    
plt.show()
plt.figure(1, figsize = (15 , 7))
plt.subplot(1 , 2 , 1)
plt.imshow(x_ray[0])
plt.title('Pneumonia')
plt.xticks([]) , plt.yticks([])

plt.subplot(1 , 2 , 2)
plt.imshow(x_ray[5000])
plt.title('Normal')
plt.xticks([]) , plt.yticks([])

plt.show()
IMAGE_SIZE = 224;
model = VGG16(include_top=False,input_shape=(224, 224, 3), weights = 'imagenet')

flat1=tf.keras.layers.Flatten()(model.output)
class1=tf.keras.layers.Dense(1024,activation='relu')(flat1)
output=tf.keras.layers.Dense(1,activation='softmax')(class1)
model=Model(model.inputs,output)
model.summary()
# Compile Mode
model.compile(loss='binary_crossentropy',
              optimizer= RMSprop(lr=2e-5),
              metrics=['acc','mse'])
# Fit Model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=5   
)
# model save
model.save_weights("vgg16-xray-pneumonia.h5")
# Visualize Loss and Accuracy Rates
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax = ax.ravel()

for i, met in enumerate(['acc', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])
