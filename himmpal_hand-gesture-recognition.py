# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.utils.np_utils import to_categorical
import random


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
imgPath = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
#         print(os.path.join(dirname, filename))
        if path.endswith('png'):
            imgPath.append(path)
            

print(len(imgPath))

 
        
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def showImage(path):
    image = cv2.imread(path)
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(img_rgb.shape)
    
    plt.imshow(img_rgb)
showImage(imgPath[2])
def createData(imgPath,val):
    images = []
    labels = []
    for path in random.choices(imgPath, k = val):
        image = cv2.imread(path)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.resize(image, (240,240))
        images.append(img_rgb)
        
        path = path.split('/')[-1]
        label = path.split('_')[2]
        labels.append(int(label))
    

    return images, labels
        
        
    
val = 20000
images, labels = createData(imgPath,val)

# print(len(img))
# print(len(lab))



images = np.array(images)
labels = np.array(labels)
images = images.reshape(val,240,240,3)
# labels = labels.tolist()
# print(labels)

onehotencoder = OneHotEncoder(sparse=False) 
labels = labels.reshape(val,1)
labels = onehotencoder.fit_transform(labels)
print(labels)

x_train,x_test,y_train,y_test =  train_test_split(images, labels, test_size = 0.2, random_state = 42)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (240,240,3)))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(128,(3,3), activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

print(model.summary())
model.compile(optimizer='RMSprop',loss='CategoricalCrossentropy', metrics = ['accuracy'])

history = model.fit(x_train,y_train, validation_data = (x_test,y_test) , epochs = 10, batch_size = 100, verbose=1)
%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()