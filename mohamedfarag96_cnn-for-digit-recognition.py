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
import numpy as np

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop
mnist = tf.keras.datasets.mnist



(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255.0

x_test =  x_test/255.0



x_train = x_train.reshape(60000,28,28,1)

x_test  = x_test.reshape(10000,28,28,1)



model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32,(5,5),activation = 'elu',input_shape = (28,28,1),padding='same'),

    tf.keras.layers.Conv2D(32,(5,5),activation = 'relu',padding='same'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64,(3,3),activation = 'elu',padding = 'same'),

    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'same'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1024,activation = 'elu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10,activation = 'softmax')

    

])







model.compile(optimizer = RMSprop(lr=0.0001),

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy'])



model.summary()



history = model.fit(x_train,y_train,epochs = 20)





train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train
train['label'].hist()
train.info()
train_data = train.loc[:, "pixel0":]

train_label= train.loc[:, "label"]



train_data = np.array(train_data)

# train_label = np.array(train_label)



train_label = tf.keras.utils.to_categorical(train_label, num_classes=10, dtype='float32')





test_data = test.loc[:, "pixel0":]

# test_label= test.loc[:, "label"]

test_data = np.array(test_data)



train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)

test_data  = test_data.reshape(test_data.shape[0],28,28,1)



train_data = train_data/255.0

test_data  = test_data/255.0



# train_datagen = ImageDataGenerator(

#                                    rotation_range = 20,

#                                    width_shift_range = 0.2,

#                                    height_shift_range = 0.2,

#                                    shear_range = 0.0,

#                                    zoom_range = 0.1,

#                                    horizontal_flip = False)

# train_datagen.fit(train_data)





model.compile(optimizer = RMSprop(lr=0.0001),

     loss='categorical_crossentropy',

     metrics=['accuracy'])



# model.summary()



history = model.fit(train_data,train_label,epochs = 25)
model.compile(optimizer = RMSprop(lr=0.0001),

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy'])



model.summary()



history = model.fit(x_test,y_test,epochs = 50)
%matplotlib inline

import matplotlib.pyplot as plt

acc = history.history['accuracy']

epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.title('Training accuracy')

plt.legend()

plt.figure()



loss = history.history['loss']

plt.plot(epochs, loss, 'b', label='Training Loss')

plt.title('Training loss')

plt.legend()



plt.show()
predictions = model.predict(test_data)

# evaluation = model.evaluate(test_data)



# print(evaluation)
prediction = []



for i in predictions:

    prediction.append(np.argmax(i))

    

    
submission =  pd.DataFrame({

        "ImageId": test.index+1,

        "Label": prediction

    })



submission.to_csv('submission.csv', index=False)
submission
