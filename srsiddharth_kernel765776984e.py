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
train_path="/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train";

test_path="/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test";

val_path="/kaggle/input/chest-xray-pneumonia/chest_xray/val";

import tensorflow as tf
import matplotlib.pyplot as plt

import PIL

a=tf.keras.preprocessing.image.load_img('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/person373_virus_756.jpeg')

plt.imshow(a)

b=tf.keras.preprocessing.image.img_to_array(a)

import numpy as np

c=np.expand_dims(b,axis=0).shape

print(c)
model=tf.keras.models.Sequential();

model.add(tf.keras.layers.Conv2D(32,(3,3),input_shape=(64,64,3),activation=tf.nn.relu))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(units=1,activation=tf.nn.sigmoid))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

val_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
training_set=train_datagen.flow_from_directory(train_path,target_size=(64,64),batch_size=32,class_mode='binary')

test_set=test_datagen.flow_from_directory(test_path,target_size=(64,64),batch_size=32,class_mode='binary')

val_set=test_datagen.flow_from_directory(val_path,target_size=(64,64),batch_size=16,class_mode='binary')



model.fit_generator(training_set,steps_per_epoch=5216/32,epochs=25,validation_data=val_set,validation_steps=16/8)
model.evaluate(test_set)
model.save('model-1.h5')
model.save_weights('weights-1.h5')
model.get_weights()
model2=tf.keras.models.Sequential();

model2.add(tf.keras.layers.Conv2D(32,(3,3),input_shape=(64,64,3),activation=tf.nn.relu))

model2.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model2.add(tf.keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu))

model2.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model2.add(tf.keras.layers.Flatten())

model2.add(tf.keras.layers.Dropout(rate=0.15))

model2.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))

model2.add(tf.keras.layers.Dropout(rate=0.15))

model2.add(tf.keras.layers.Dense(units=1,activation=tf.nn.sigmoid))
model2.load_weights('weights-1.h5')
model2.get_weights()
model2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model2.fit_generator(training_set,steps_per_epoch=5216/32,epochs=25,validation_data=val_set,validation_steps=16/8)
model2.evaluate(test_set)
model2.save('model-2.h5')

model2.save_weights('model-2.h5')
model2_json=model2.to_json()
model3=tf.keras.models.model_from_json(model2_json)
model3.summary()
model3.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model3.fit_generator(training_set,steps_per_epoch=5216/32,epochs=25,validation_data=val_set,validation_steps=16/16)
model3.evaluate(test_set)
model3.save('model-3.h5')

model3.save_weights('weights-3.h5')