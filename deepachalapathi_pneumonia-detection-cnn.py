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
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                shear_range = 0.2, zoom_range = 0.2,
                                rotation_range = 30,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                horizontal_flip = True, 
                                vertical_flip=False)

training_set = train_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/chest_xray/train', target_size = (100, 100), batch_size = 64, class_mode = 'binary')
test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/chest_xray/test', target_size = (100, 100), batch_size = 64, class_mode = 'binary')
val_datagen = ImageDataGenerator(rescale = 1./255)

val_set =val_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/chest_xray/val', target_size = (100, 100), batch_size = 64, class_mode = 'binary')
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters =32, kernel_size= (3,3), input_shape=[100, 100, 3]))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size= (3,3) , activation='relu'))

model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#increasing the number of epcochs might make the model much better
model.fit(x = training_set, validation_data =val_set, epochs = 10)
print("Loss of the model is  " , model.evaluate(test_set)[0])
print("Accuracy of the model is  " , model.evaluate(test_set)[1]*100 , "%")
file_normal = os.listdir('../input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL')
normal = len(file_normal)
print("Number of Files using listdir method#1 :", normal)
file_diseased = os.listdir('../input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA')
diseased = len(file_diseased)
print("Number of Files using listdir method#1 :", diseased)
y_test = test_set.classes
print(y_test)
print(test_set.class_indices)
y_pred = model.predict_classes(test_set)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test, y_pred, target_names = ['Normal (Class 0)', 'Pneumonia (Class 1)']))
cm = confusion_matrix(y_test,y_pred)
print(cm)