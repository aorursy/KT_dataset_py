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
from keras.layers import Input,Lambda,Dense,Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
train_path="/kaggle/input/chest-xray-pneumonia/chest_xray/train/"
train_path="/kaggle/input/chest-xray-pneumonia/chest_xray/test/"
vgg=VGG16(input_shape=[224,224,3],weights='imagenet',include_top=False)
for layer in vgg.layers:
    layer.trainable=False
x=Flatten()(vgg.output)
x=Dense(512,activation='relu')(x)
prediction=Dense(2,activation='softmax')(x)
model=Model(inputs=vgg.input,output=prediction)
model.summary()

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
initial_learning_rate=0.005
lr_schedule=tf.keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate=initial_learning_rate,
    decay_steps=5,
    decay_rate=0.96,
    staircase=True
)
optimizer=Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1.255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
train_set=train_datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/train/',
                                           target_size=(224,224),batch_size=32,class_mode='categorical')
test_set=test_datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/test/',
                                         target_size=(224,224),batch_size=32,class_mode='categorical')


!pip install livelossplot
from livelossplot.tf_keras import PlotLossesCallback
callbacks=[PlotLossesCallback()]
history=model.fit(train_set,steps_per_epoch=len(train_set)//32,validation_data=test_set,validation_steps=len(test_set)//32,epochs=7,callbacks=callbacks)
plt.plot(history.history['accuracy'],label="train_acc")
plt.plot(history.history['val_accuracy'],label='test_acc')
plt.legend()
plt.show()

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
img=image.load_img('/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg',target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
classes=model.predict(img_data)
print(classes)