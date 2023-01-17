import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

l = []

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        l.append([os.path.join(dirname,filename),int(dirname=="/kaggle/input/damage_crop")])
df = pd.DataFrame(l,columns = ['path','label'])
df.head()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf 

import keras 

from keras.models import Sequential

from keras import layers

from keras.layers import Dense,Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, InputLayer

import cv2

import random

import PIL.Image as Image 

from sklearn.model_selection import train_test_split

np.random.seed(0)
def resize_image(image_array):

    return cv2.resize(image_array,(224,224))

def read_image(image_path):

    return cv2.imread(image_path)

def rescale_image(image_array):

    return image_array*1./255

def preprocess_image(image_path,reshape = True,read=True):

    if(read==True):

        image = read_image(image_path)

    image = resize_image(image)

    image = rescale_image(image)

    if(reshape ==  True):

        image = image.reshape(-1,image.shape[0],image.shape[1],image.shape[2])    

    return image
images = []

for n,i in enumerate(df.path.values):

    print(i)

    try:

        images.append(preprocess_image(i,reshape=False))

    except:

        pd.DataFrame(df, index = [n])

print(images[0])        
images = np.array(images)
images.shape


def build_model():

    keras.backend.clear_session()

    vgg = keras.applications.VGG16(input_shape=(224,224,3),include_top=False,weights='imagenet',pooling='avg')

    vgg.trainable=False

    vggmodel = keras.Sequential([vgg

                             ,Dense(220,activation='relu'),Dense(220,activation='relu'),Dense(1,activation='sigmoid')])



    vggmodel.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])

    

  

    

    return vggmodel
model = build_model()
print(model.input_shape)

print(model.output_shape)
labels = df.label.values

labels = labels.reshape(-1,1)
hist = model.fit(images,labels,epochs=25,validation_split=0.2)
plt.figure(figsize=(10,7))

plt.subplot(1,2,1)

plt.plot(hist.history['accuracy'],label='accuracy')

plt.plot(hist.history['loss'],label='loss')

plt.legend()

plt.title("training set")

plt.grid()

plt.subplot(1,2,2)

plt.plot(hist.history['val_accuracy'],label='val_accuracy')

plt.plot(hist.history['val_loss'],label='val_loss')

plt.legend()

plt.title("validation set")

plt.grid()

plt.ylim((0,4))
model.save_weights('floodmodel.h5')
ig = plt.imread(df['path'][0])

plt.imshow(ig)
newimg = preprocess_image(df['path'][0])

print(model.predict(newimg))

print(int(model.predict(newimg)[0][0]>0.5))
newimg = preprocess_image(df['path'][49])

plt.imshow(newimg[0])

print(model.predict(newimg))

print(int(model.predict(newimg)[0][0]>0.5))
for i in df.path.values:

    try:

        print(int(model.predict(preprocess_image(i))[0]>0.5))

    except:

        pass