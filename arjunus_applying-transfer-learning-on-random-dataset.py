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
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
#importing other required libraries
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import unique_labels
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import Sequential
from keras.applications import VGG19 #For Transfer Learning
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from keras.utils import to_categorical
url = '/kaggle/input/natural-images/natural_images'
train_data = tf.keras.preprocessing.image_dataset_from_directory(url,batch_size=32,image_size=(128,128), shuffle=True)
class_name = train_data.class_names
print(class_name)
tf.data.experimental.cardinality(train_data).numpy()
for img,lab in  train_data.take(1):
    print(img.shape)
    print(lab.shape)
    plt.figure(figsize=(20,20))
    for i in range(32):
        plt.subplot(6,6,i+1)
        plt.imshow(img[i]/255.0)
        plt.axis('off')
        plt.title(class_name[lab[i]])
#from keras.preprocessing.image import ImageDataGenerator
"""train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)

val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)"""
test_data = train_data.take(30)
train_data = train_data.skip(30)

val_data = test_data.take(15)
test_data = test_data.skip(15)

print("train_data_shape",tf.data.experimental.cardinality(train_data).numpy())
print("val_data_shape",tf.data.experimental.cardinality(val_data).numpy())
print("test_data_shape",tf.data.experimental.cardinality(test_data).numpy())
"""data =[]
label =[]

for img,lab in train_data:
    data.append(img)
    label.append(lab)
t_data = np.concatenate(data)
t_label = np.concatenate(label)"""
"""v_data =[]
v_label =[]

for img,lab in val_data:
    v_data.append(img)
    v_label.append(lab)
vt_data = np.concatenate(v_data)
vt_label = np.concatenate(v_label)"""
#vt_label = tf.keras.utils.to_categorical(vt_label,num_classes=8)
#t_label = tf.keras.utils.to_categorical(t_label,num_classes=8)#
"""print(t_data.shape,t_label.shape)
print(vt_data.shape,vt_label.shape)"""
"""train_generator.fit(t_data)
val_generator.fit(vt_data)"""
process_unit = tf.keras.applications.resnet50.preprocess_input
#base_model = tf.keras.applications.ResNet50()
#base_model.summary()
base_model = tf.keras.applications.ResNet50(input_shape=(128,128,3),include_top=False,weights='imagenet')
base_model.trainable = False
inputs = tf.keras.Input(shape=(128,128,3))
x = base_model(inputs,training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024,activation='relu')(x)
outputs = tf.keras.layers.Dense(8,activation='softmax')(x)

n_model = tf.keras.Model(inputs,outputs)
n_model.summary()
n_model.compile(optimizer=tf.keras.optimizers.Nadam(),loss =tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'] )
hist = n_model.fit(train_data,epochs=5,validation_data=val_data,verbose=2)
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(train_acc,label='training accuracy')
plt.plot(val_acc,label='validation accuracy')
plt.legend()
plt.title('Accuracy graph')
plt.subplot(2,1,2)
plt.plot(train_loss,label='training loss')
plt.plot(val_loss,label='validation loss')
plt.legend()
plt.title('Loss graph')

