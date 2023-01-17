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

import os

import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import tensorflow as tf
from tqdm import tqdm

%matplotlib inline
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
print(train_df.shape)
print(test_df.shape)
train_df.head()
def preprocess_image(image_path, desired_size=224):
    im = Image.open(image_path)
    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    
    return im
N = train_df.shape[0]
x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(train_df['id_code'])):
    x_train[i, :, :, :] = preprocess_image(
        f'../input/aptos2019-blindness-detection/train_images/{image_id}.png'
    )
N = test_df.shape[0]
x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(test_df['id_code'])):
    x_test[i, :, :, :] = preprocess_image(
        f'../input/aptos2019-blindness-detection/test_images/{image_id}.png'
    )
y_train = pd.get_dummies(train_df['diagnosis']).values

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
y_train_multi[:, 4] = y_train[:, 4]

for i in range(3, -1, -1):
    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])

print("Original y_train:", y_train.sum(axis=0))
print("Multilabel version:", y_train_multi.sum(axis=0))
fig, axs = plt.subplots(1,5,figsize=(15, 4),sharey=True)
for i,item in enumerate(range(5)):
    axs[i].imshow(x_train[i])
    axs[i].set_title(y_train[i])
x_train2=[]
# image processing , here we processing the images and we using blending technique 
for image in x_train:
    img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    processed_image=cv2.addWeighted ( img,4, cv2.GaussianBlur( img, (0,0) , 250/10) ,-4 ,128) # blending technique 
    x_train2.append(processed_image)
fig, axs = plt.subplots(1,5,figsize=(15, 4),sharey=True)
for i,item in enumerate(range(5)):
    axs[i].imshow(x_train2[i])
    axs[i].set_title(y_train[i])
x_train2=np.array(x_train2)  # converting into numpy array 
#splitting the data 
X_train, X_val, y_train, y_val = train_test_split(
    x_train2, y_train_multi, 
    test_size=0.15, 
    random_state=2019
)
x_train2.shape
y_train_multi.shape
X_train.shape
y_train.shape
#we can also import the resnet 
densenet = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)
def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2024,activation="relu"))  # we can increase the layer for higher accuracy 
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )
    
    return model
model = build_model()
model.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator    # data augmentation 

train_datagen = ImageDataGenerator(
      #rotation_range=30,
      shear_range=0.1,
      zoom_range=[0.3,0.5],
      #width_shift_range=0.4,
      #height_shift_range=0.4,
      horizontal_flip=True,
      vertical_flip=True,
  
      fill_mode='nearest')


test_datagen=ImageDataGenerator()
train_set=train_datagen.flow(X_train,y_train)
test_set=test_datagen.flow(X_val,y_val)
history=model.fit_generator(
         train_set,
        validation_data=test_set,
        epochs=5,
        verbose=1
     )
model.save("messidor_analyzer.h5")
model.evaluate(X_val,y_val)
pred=model.predict(X_val)
pred=np.sum(pred,axis=1)     # summing the prediction so that we can see the output 
pred

