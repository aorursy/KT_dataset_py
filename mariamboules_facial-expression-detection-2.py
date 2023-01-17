# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# !pip install utils
# import utils

import csv
from PIL import Image    
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np # linear algebra
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
from keras.layers import Dense, Activation
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
X_train,X_test,Y_train,Y_test = train_test_split(data_set.pixels,data_set.emotion, test_size=0.2)
#print(X_train[11])

oversample = RandomOverSampler(sampling_strategy='auto')
# fit and apply the transform
X_over, y_over = oversample.fit_resample((X_train).values.reshape(-1, 1), Y_train)

a = np.array(y_over)
collections.Counter(a)

y_over = pd.Series(y_over)
y_over= y_over.values.reshape(len(y_over),1)
def preprocessing(pixels):
    a = []
    for i in range(len(data_set)):
        try:
            image_string = (pixels)[i].split(' ') 
            image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48,1)
            a.append(image_data)
        except:
            i+=1
    return a
%matplotlib inline
a= []
#print(X_over)
X_over = pd.Series(X_over.flatten())

print(X_over)
a = preprocessing(X_over)

X_train = np.array(a)
#X_test =test
Y_train = y_over

print ("number of training examples = " + str(X_train.shape[0]))
#print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
# print ("X_test shape: " + str(X_test.shape))


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

model.fit(X_train, Y_train, batch_size=64, epochs=10, steps_per_epoch=(len(X_train)/128), validation_split = 0.25)

model.save('CNNmodel')
#print(X_test.pixels[0])
Expressions = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
a= []
a = preprocessing(X_test)
      

X_testing_array = np.array(a)
prediction = model.predict(X_testing_array)
model.evaluate(X_testing_array,Y_test)
results = Expressions[np.argmax(prediction[100])]
print(results)
