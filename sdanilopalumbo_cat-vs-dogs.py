import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import pickle
import time


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.client import device_lib
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/cat-and-dog/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATADIR = "../input/cat-and-dog/training_set/training_set"
CATEGORIES = ['cats','dogs']
for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap = 'gray')
        break
    break
IMG_SIZE = 70

new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_array, cmap='gray')
training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass    
create_training_data()
print(len(training_data))
import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])
X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE,IMG_SIZE,1)
pickle_out = open('/kaggle/working/X.pickle','wb')
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open('/kaggle/working/y.pickle','wb')
pickle.dump(y,pickle_out)
pickle_out.close()
X = pickle.load(open('/kaggle/working/X.pickle','rb'))
y = pickle.load(open('/kaggle/working/y.pickle','rb'))
X = X/255.0
y = np.array(y)
NAME = 'Cats-vs-dogs_64x2-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
## NEURAL NETWORK

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, epochs=45, validation_split=0.3)
model.summary()
print('Loss: ',model.evaluate(X,y)[0] +'\n' + 'Accuracy: ', model.evaluate(X,y)[1])
model.save('/kaggle/working/model.model')
CATEGORIES = ["cats", "dogs"]  # will use this to convert prediction num to string value
model = tf.keras.models.load_model("/kaggle/working/model.model")
def prepare(filepath):
    IMG_SIZE = 70  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    plt.imshow(img_array, cmap = 'gray')
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.
training_data_cats = []

for dirname, _, filenames in os.walk('/kaggle/input/cat-and-dog/training_set/training_set/cats/'):
    for filename in filenames:
        name = os.path.join(dirname,filename)
        training_data_cats.append(name)
training_data_dogs = []

for dirname, _, filenames in os.walk('/kaggle/input/cat-and-dog/training_set/training_set/dogs/'):
    for filename in filenames:
        name = os.path.join(dirname,filename)
        training_data_dogs.append(name)
training_data_cats
training_data_dogs
training_data_cats = training_data_cats[:10]
training_data_dogs = training_data_dogs[:10]
training_data_cats
training_data_dogs
prediction = model.predict([prepare(training_data_cats[3])])  
print(CATEGORIES[int(prediction[0][0])])