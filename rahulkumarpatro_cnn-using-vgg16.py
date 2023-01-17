# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    
        print(os.path.join(dirname))

# Any results you write to the current directory are saved as output.
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import random
filenames=os.listdir("/kaggle/input/dogs-vs-cats/train/train")
# print(filenames)
category=[]

for i in filenames:
    catg=i.split('.')[0]
#     print(catg)

    if catg=='cat':
        category.append('0') 
        
    else:
        category.append('1')    
category[0:5]
df=pd.DataFrame({'filename': filenames,
    'category': category})
df.tail()
sample = random.choice(filenames)
image = load_img("/kaggle/input/dogs-vs-cats/train/train/"+sample)
plt.imshow(image)
image.size
from keras.models import Sequential
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation,GlobalMaxPooling2D
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications import VGG16
from keras.models import Model
img_size = [224, 224,3]
vgg = VGG16(input_shape=img_size, weights='imagenet', include_top=False)

for layer in vgg.layers[:15]:
    layer.trainable = False

for layer in vgg.layers[15:]:
    layer.trainable = True
    
last_layer = vgg.get_layer('block5_pool')
last_output = last_layer.output
    

x = GlobalMaxPooling2D()(last_output)

x = Dense(512, activation='relu')(x)

x = Dropout(0.5)(x)

x = layers.Dense(1, activation='sigmoid')(x)

model = Model(vgg.input, x)

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# x = Flatten()(vgg.output)
# prediction = Dense(1, activation='sigmoid')(x)
# model = Model(inputs=vgg.input, outputs=prediction)


model.summary()
model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
train_df, test_df = train_test_split(df, test_size=0.1)
train_df = train_df.reset_index()
test_df = test_df.reset_index()


total_train = train_df.shape[0]
total_test = test_df.shape[0]
print(total_train)
print(total_test)
df.dtypes
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "/kaggle/input/dogs-vs-cats/train/train/", 
    x_col='filename',
    y_col='category',
    class_mode='binary',
    target_size=(224,224),
    batch_size=32
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_generator = validation_datagen.flow_from_dataframe(
    test_df, 
    "/kaggle/input/dogs-vs-cats/train/train/", 
    x_col='filename',
    y_col='category',
    class_mode='binary',
    target_size=(224,224),
    batch_size=32
)
len(train_generator)
len(validation_generator)
r = model.fit_generator(
  train_generator,
  validation_data=test_generator,
  epochs=5,
  steps_per_epoch=704,
  validation_steps=79
)
loss, accuracy = model.evaluate_generator(test_generator, total_test//32)
print("Test: accuracy = %f  ;  loss = %f " % (accuracy, loss))
import tensorflow as tf

from keras.models import load_model
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()


model.save('model_final.h5')