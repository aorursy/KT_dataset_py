# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten,Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential

import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
class_name = ["T-shirt","Trouser","Pullover","Dress","Coat","Sandal", "Shirt","Sneaker","Bag","Boot"]
#6000 images of pixel 28,28
print("train images shape",train_images.shape)
print("test images shape",test_images.shape)
#create validation set
train_images = train_images[5000:]
train_labels = train_labels[5000:]

val_images = train_images[:5000]
val_labels = train_labels[:5000]
train_images.shape
for i in range(0,5):
   plt.imshow(val_images[i], cmap='Greys')
   plt.title(class_name[val_labels[i]]) 
   plt.axis('off')
   plt.show()
sns.countplot(train_labels)
sns.countplot(val_labels)
sns.countplot(test_labels)
train_images = train_images.reshape(55000,28,28,1)
val_images= val_images.reshape(5000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)
train_images = train_images/255
val_images = val_images/255
test_images = test_images/255
model = Sequential()
model.add(Conv2D(64,7,activation='relu', padding='same',input_shape=[28,28,1]))
model.add(MaxPooling2D(2))
model.add(Conv2D(128,3,activation='relu', padding='same'))
model.add(Conv2D(128,3,activation='relu', padding='same'))
model.add(MaxPooling2D(2))
model.add(Conv2D(256,3,activation='relu', padding='same'))
model.add(Conv2D(256,3,activation='relu', padding='same'))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.summary()
model.layers
model.layers[1].name
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model.fit(train_images,train_labels,epochs=30,validation_data=(val_images,val_labels))
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
pd.DataFrame(history.history).boxplot()
model.evaluate(test_images,test_labels, verbose=0)
y_preds = model.predict_classes(test_images)
confusion_matrix(test_labels,y_preds)

