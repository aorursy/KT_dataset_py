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
# Importing necessary libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import random
# Checking the training data
file_train = "/kaggle/input/sign-language-mnist/sign_mnist_train.csv"
data_train = pd.read_csv(file_train)
# data length
print(data_train.shape)
# data.describe
#print(data_train.describe())

# Checking the test data
file_test = "/kaggle/input/sign-language-mnist/sign_mnist_test.csv"
data_test = pd.read_csv(file_test)
#data shape
print(data_test.shape)
# Preprocess Function
  # Convert the labels into categorical forms i.e telling when a particular label has occured or not
  # Selecting the image pixel values and normalizing them and reshapping them
  # Returning the labels , images
def preprocess(data):
    num_classes = 25
    num_images = data.shape[0]
    target = tf.keras.utils.to_categorical(data.label,num_classes)
    images_p = (data.values[:,1:])
    images_p1 = images_p.reshape(num_images,28,28,1)
    return images_p1,target

train_X,train_y = preprocess(data_train)
test_X,test_y   = preprocess(data_test)
plt.figure(figsize = (7,7))
rand = random.randrange(0,data_train.shape[0])
plt.imshow(train_X[rand][:,:,-1],cmap="gray")
plt.show()
plt.figure(figsize=(7,7))
rand = random.randrange(0,test_X.shape[0])
plt.imshow(test_X[rand][:,:,0],cmap="inferno")
plt.show()
# Building a model  using Convolutional layers whch will use features to find features 
# Max Pooling layers to compress the image by making sure that the features remain intact
# Dropout -- to reduce overfitting by randomly making a couple weight values as zero
num_classes = 25
model_classifier = tf.keras.models.Sequential([
                   tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation="relu",input_shape=(28,28,1)),
                   tf.keras.layers.MaxPooling2D(2,2),
                   tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation="relu"),
                   tf.keras.layers.MaxPooling2D(2,2),
                   tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation="relu"),
                   tf.keras.layers.MaxPooling2D(2,2),
                   tf.keras.layers.Dropout(0.2),
                   tf.keras.layers.Flatten(),
                   tf.keras.layers.Dense(512,activation="relu"),
                   tf.keras.layers.Dense(num_classes,activation="softmax")
    
])
# The journey of the image through the model for classification purpose
model_classifier.summary()
model_classifier.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.001),loss = "categorical_crossentropy" ,
                         metrics =['acc'])
history1 = model_classifier.fit(train_X,train_y,epochs=10, steps_per_epoch = train_X.shape[0]/16, validation_split=0.3)
model_classifier.evaluate(test_X,test_y)
model_classifier.compile(optimizer="adam" , loss = "categorical_crossentropy" , metrics = ['acc'])
history2 = model_classifier.fit(train_X,train_y,epochs = 10,validation_split=0.2)
model_classifier.evaluate(test_X,test_y)
# the training and testing accuracy and respective losses
plt.figure()
acc      = history1.history['acc' ]
val_acc  = history1.history['val_acc']
loss     = history1.history['loss']
val_loss = history1.history['val_loss']
sns.lineplot(x = [i for i in range(1,len(acc)+1)],y = acc)
sns.lineplot(x = [i for i in range(1,len(val_acc)+1)],y=val_acc)


plt.figure()
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
# the testing accuracy , val_acc , loss, val_loss
plt.figure()
acc = history2.history['acc']
val_acc = history2.history['val_acc']
plt.plot(acc)
plt.plot(val_acc)

plt.figure()
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
# Using Image Data Generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rescale = 1./255,
                              rotation_range=40,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              fill_mode="nearest")

val_gen = ImageDataGenerator(rescale=1./255)
history3 = model_classifier.fit(train_gen.flow(train_X,train_y),
            epochs = 10,
           validation_data=val_gen.flow(test_X,test_y))
plt.figure()
plt.plot(history3.history['acc'])
plt.plot(history3.history['val_acc'])
plt.figure()
plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
