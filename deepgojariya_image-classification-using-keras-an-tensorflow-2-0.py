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
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
training_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.3,zoom_range=0.3,horizontal_flip=True)
train_set = training_datagen.flow_from_directory('/kaggle/input/intel-image-classification/seg_train/seg_train',target_size=(150,150),batch_size=32,class_mode='categorical')
test_datagen = ImageDataGenerator(rescale=1./255)
validation_set = test_datagen.flow_from_directory('/kaggle/input/intel-image-classification/seg_test/seg_test',target_size=(150,150),batch_size=32,class_mode='categorical')
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',input_shape=[150,150,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=3,strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=3,strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=3,strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=200,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=178,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=200,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=6,activation='softmax'))
cnn.summary()
cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
cnn.fit(x=train_set,validation_data = validation_set,epochs=30)
cnn.save('model.h5')
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/kaggle/input/intel-image-classification/seg_pred/seg_pred/101.jpg', target_size = (150,150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
result=list(result[0])
print(result.index(max(result)))
print(result)
img_index = result.index(max(result))
print(opDict[img_index])
train_set.class_indices
loaded_model = tf.keras.models.load_model('model.h5')
opDict = {0:'Buildings',1:'Forests',2:'Glaciers',3:'Mountains',4:'Sea',5:'Street'}
