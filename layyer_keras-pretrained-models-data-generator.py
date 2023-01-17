# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

'''

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

'''
import numpy as np

import tensorflow as tf

from matplotlib import pyplot as plt

import os

import pandas as pd
train_label = pd.read_csv('/kaggle/input/super-ai-image-classification/train/train/train.csv')
train_label.head()
train_label.category.plot.hist()
class0 = train_label[train_label.category == 0]

class1 = train_label[train_label.category == 1]
os.makedirs('train_data')
from shutil import copyfile
os.makedirs('train_data/class0')

for file in class0.id:

    src = "/kaggle/input/super-ai-image-classification/train/train/images/"+file

    des = 'train_data/class0/'+file

    copyfile(src,des)
os.makedirs('train_data/class1')

for file in class1.id:

    src = "/kaggle/input/super-ai-image-classification/train/train/images/"+file

    des = 'train_data/class1/'+file

    copyfile(src,des)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    validation_split=0.30,

    rescale=1./255,

    shear_range=0.2,

    zoom_range=0.3,

)
train_generator = datagen.flow_from_directory('./train_data',target_size=(224,224),batch_size=32,class_mode="binary",subset='training') 

test_generator = datagen.flow_from_directory('./train_data',target_size=(224,224),batch_size=32,class_mode="binary",subset='validation',shuffle=False) 
pretrain = tf.keras.applications.MobileNetV2(weights='imagenet',include_top=False,input_shape=(224,224,3))
pretrain.trainable = False
last_pretrain = pretrain.layers[-1].output

x = tf.keras.layers.GlobalAveragePooling2D()(last_pretrain)

x = tf.keras.layers.Dropout(0.25)(x)

x = tf.keras.layers.Dense(128,activation='relu')(x)

x = tf.keras.layers.Dropout(0.25)(x)

x =  tf.keras.layers.Dense(1,activation='sigmoid')(x)

model = tf.keras.Model(pretrain.input,x)
model.summary()
model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(),metrics=['acc'])
#since data is imbalance

class_weight = {0:1,1:3}
history = model.fit(train_generator ,validation_data=test_generator,

                    epochs=10,steps_per_epoch=len(train_generator),

                    class_weight=class_weight)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])
model.save('super_ai_hw3_1.h5')
from sklearn.metrics import classification_report
pred = (model.predict_generator(test_generator) > 0.5).astype(int).reshape(-1)
pred.shape,test_generator.labels.shape
print(classification_report(test_generator.labels,pred))
(pred==test_generator.labels).sum()/len(pred)
pred_datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    rescale=1./255

)
val_gen = datagen.flow_from_directory('/kaggle/input/super-ai-image-classification/val/val/',target_size=(224,224),batch_size=32,shuffle=False,class_mode="binary") 
pred = model.predict_generator(val_gen)
pred = (pred>0.5).astype(int).reshape(-1)
val_name = [x.split('/')[1] for x in val_gen.filenames]
val = pd.DataFrame()

val["id"] = val_name

val["category"] = pred
val.head()
val.to_csv('val_submit.csv', index=False)