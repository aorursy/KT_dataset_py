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
import numpy as np

import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import keras

import tensorflow as tf

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

dig_minst =pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")

print("data is here ")
x_train=train.iloc[:,1:].values 

y_train=train.iloc[:,0].values 

y_train[:10]
import random

for i in range(3):

    image = x_train[random.randint(0,1024)].reshape(28,28)

    plt.imshow(image, cmap="gray")

    plt.axis("off")

    plt.show()
x_train =x_train.reshape(-1,28,28,1)

y_train =keras.utils.to_categorical(y_train,num_classes =10)

print("shape of x_train :\t ",x_train.shape)

print("shape of y_train :\t ",y_train.shape)




train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')



validation_datagen = ImageDataGenerator(

    rescale=1. / 255)



x_train ,x_validation ,y_train ,y_validation= train_test_split(x_train,y_train,test_size =.2)


model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])



model.compile(optimizer = 'adam',

              loss = 'categorical_crossentropy',

              metrics=['accuracy'])

model.summary()






history = model.fit(

    x_train,y_train ,batch_size=32,validation_data=[x_validation,y_validation],

      steps_per_epoch=len(x_train)/32,epochs = 10)

    

    



      

               

    
dig_minst.head()

x_dig_mnist =dig_minst.drop('label',axis =1).iloc[:,:].values

x_dig_mnist =x_dig_mnist.reshape(-1,28,28,1)

y_dig_mnist = dig_minst['label']

y_dig_mnist =keras.utils.to_categorical(y_dig_mnist,10)

model.evaluate(x_dig_mnist,y_dig_mnist)
x_test =np.array(test.drop("id",axis =1).iloc[:,:].values)

x_test = x_test.reshape(x_test.shape[0], 28, 28,1)/255
y_prob = model.predict(x_test) 

y_classes = y_prob.argmax(axis=-1)

y_classes


acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label']=y_classes

submission.to_csv("submission.csv",index=False)