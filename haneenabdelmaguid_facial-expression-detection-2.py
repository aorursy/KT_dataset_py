# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras
import csv
from PIL import Image    
from sklearn.model_selection import train_test_split
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D,BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tqdm import tqdm
import numpy as np # linear algebra
from numpy import asarray
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
import collections
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# from tensorflow.keras.utils import plot_model
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        print(os.listdir("../input"))
        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


data_set = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/train.csv')
examples = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/test.csv')
icml_face_data = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv')
ax = np.array(data_set.emotion)
collections.Counter(ax)
oversample = RandomOverSampler(sampling_strategy='auto')
# fit and apply the transform
X_over, y_over = oversample.fit_resample((data_set.pixels).values.reshape(-1, 1), data_set.emotion)


a = np.array(y_over)
collections.Counter(a)






y_over = pd.Series(y_over)
y_over= y_over.values.reshape(len(y_over),1)

X_train,X_test,Y_train,Y_test = train_test_split(X_over,y_over, test_size=0.2)
#print(X_train[11])
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

def preprocessing(pixels):
    a = []
    
    for i in range(len(pixels)):
            image_string = (pixels)[i].split(' ') 
            image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48,1)
            a.append(image_data)

    return a
%matplotlib inline
a= []

X_train = pd.Series(X_train.flatten())
X_train_resnet50 =X_train
a = preprocessing(X_train)



X_train = np.array(a)
#X_test =test
# Y_train = y_over

print ("number of training examples = " + str(X_train.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))


# ResNet50
a = []
for i in range(len(X_train_resnet50)):
            image_string = (X_train_resnet50)[i].split(' ') 
            image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
            a.append(image_data)

X_train_resnet50 = np.array(a)
rgb_X_train = np.repeat(X_train_resnet50[..., np.newaxis], 3, -1)
print(rgb_X_train.shape)  # (size, 48, 48, 3)

model1 = ResNet50(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

x = model1.output
x= Flatten()(x)
x = Dense(7, activation='softmax')(x)
model50 = Model(inputs=model1.input, outputs=x)


model50.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model50.fit(rgb_X_train, Y_train, batch_size=64, epochs=40, steps_per_epoch=len(X_train)/128, validation_split = 0.25)
model50.save('CNNmodel')

Expressions = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
a= []
print ("X_test shape: " + str(X_test.shape))
X_test50 = pd.Series(X_test.flatten())
a = []
for i in range(len(X_test50)):
            image_string = (X_test50)[i].split(' ') 
            image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
            a.append(image_data)


X_testing_array = np.array(a)
print(X_testing_array.shape)  # (size, 48, 48)
rgb_X_test = np.repeat(X_testing_array[..., np.newaxis], 3, -1)
# print(rgb_X_test.shape)  # (size, 48, 48, 3)

prediction = model50.predict(rgb_X_test)
model50.evaluate(rgb_X_test,Y_test)
    
results = Expressions[np.argmax(prediction[100])]
print(results)

image_string = X_test50[100].split(' ') 
image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
plt.imshow(image_data)
model = models.Sequential()
model.add(layers.Conv2D(64, (1, 1), padding='same', activation='relu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(layers.Conv2D(128, (3, 3),padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(layers.Conv2D(256, (5, 5),padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2, 2),padding="same"))
model.add(Dropout(0.25))


model.add(layers.Flatten())

model.add(layers.Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(layers.Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(layers.Dense(7, activation='softmax'))
model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.fit(X_train, Y_train, batch_size=64, epochs=40, steps_per_epoch=(len(X_train)/128))


model.fit(X_train, Y_train, batch_size=64, epochs=40, steps_per_epoch=len(X_train)/128, validation_split = 0.25)

model.save('CNNmodel')

Expressions = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
a= []
print ("X_test shape: " + str(X_test.shape))
X_test = pd.Series(X_test.flatten())

a = preprocessing(X_test)
X_testing_array = np.array(a)


print ("number of Test examples = " + str(X_testing_array.shape[0]))
print ("X_test shape: " + str(X_testing_array.shape))
print ("Y_test shape: " + str(Y_test.shape))

prediction = model.predict(X_testing_array)
model.evaluate(X_testing_array,Y_test)

results = Expressions[np.argmax(prediction[100])]
print(results)
image_string = X_test[100].split(' ') 
image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
plt.imshow(image_data)