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
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from pprint import pprint
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from numpy import expand_dims
import tensorflow.keras.layers as Layers
# model
from tensorflow.keras.models import Sequential

# layers
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Activation, Dropout

# categorical
from tensorflow.keras.utils import to_categorical
import os
import glob
def dataset_create(path):
    """reading images in each folder and creating our data so the model 
    to be trained on"""
    images = []
    labels = []
    for folder in os.listdir(path):
        image_path = os.path.join(path, folder)
        for filename in glob.glob(image_path + '/*.jpg'):
            image = cv.imread(filename)
            image = cv.resize(image,(150,150))
            images.append(image)
            labels.append(folder)
    return np.array(images), np.array(labels)
path_train = '/kaggle/input/intel-image-classification/seg_train/seg_train'

x_train, y_train = dataset_create(path_train)
x_train.shape, y_train.shape

import seaborn as sns
sns.set(rc = {'figure.figsize':(12,7)})
p = sns.countplot( x = y_train, palette = 'Set3')

#I think we can consider our training dataset as balanced dataset


#converting the lables to the categorical number they should have:
def y_label(y):
    label_categories = {'buildings': 0,
            'forest': 1,
            'glacier': 2,
            'mountain': 3,
            'sea': 4,
            'street': 5}

#y_train_cat = map(label_categories.get, y_train_array)
    y = [label_categories[k] for k in y]
    return y

y_train = y_label(y_train)

import random
def shuffle_data(x,y):
    c = list(zip(x, y))
    random.shuffle(c)
    x, y = zip(*c)
    x= np.asarray(x)
    y= np.asarray(y)
    return x, y

x_train, y_train = shuffle_data(x_train,y_train)
type(y_train)
y_train = to_categorical(y_train)
# define the model
model = Sequential([
    # First convolutional layer
    Conv2D(filters= 64, kernel_size=(3, 3), input_shape=(150, 150, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Activation('relu'),
    # Second convolutional layer
    #Conv2D(filters= 16, kernel_size=(3, 3),input_shape=(150, 150, 3)),
    #MaxPool2D(pool_size=(2, 2), strides=2),
    #Activation('relu'),
    # Fully connected layers
    Flatten(),#before that we need to flatten our inputs
    #Dense(units =128, activation = 'relu'),
    #Dropout(0.5),
    Dense(units =64, activation = 'relu'),
    #Dropout(0.2),
    Dense(units =32, activation = 'relu'),
    #Dropout(0.2),
    Dense(units=6, activation='softmax') # like binary Logistic Regression classifier, Softmax classifier is its generalization to multiple classes (here we have 6).
])
model.summary()
# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy']
)
history = model.fit(x=x_train, y=y_train, batch_size=128, epochs=20, validation_split=0.2)


#from tensorflow.keras import backend as K

#K.clear_session()




# plot the accuracy
plt.figure(figsize=(12,4))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()


# now lets test it on our test data
# firts we need to prepare our test data
path_test = '/kaggle/input/intel-image-classification/seg_test/seg_test'

x_test, y_test = dataset_create(path_test)
y_test = y_label(y_test)
x_test, y_test = shuffle_data(x_test,y_test)
y_test = to_categorical(y_test)
score = model.evaluate(x_test, y_test)
print(score)
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
model_2 = VGG16(weights='imagenet')

model_2 = VGG16(input_shape=(150, 150, 3), weights='imagenet', include_top=False)
# freeze the layers in Convolutional part so that do not retrain them
for layer in model_2.layers:
    layer.trainable = False
    print(layer.trainable)


# Now lets add the fully connected part as we like to our model_2
edited_model = Sequential([
    model_2,
    Flatten(),
    Dense(16, activation='relu'),
    Dense(6, activation='softmax')
])


# compile the model
edited_model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy']
)
history_2 = edited_model.fit(x=x_train, y=y_train, batch_size=32, epochs=20, validation_split=0.2)
