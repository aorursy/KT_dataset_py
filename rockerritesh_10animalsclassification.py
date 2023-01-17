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
DATADIR = "/kaggle/input/animals10/raw-img"
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATADIR = "/kaggle/input/animals10/raw-img"
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
CATEGORIES = ["farfalla", "scoiattolo","pecora","gallina","mucca","elefante","cane","ragno","cavallo","gatto"]
for category in CATEGORIES:  # do in all image
    path = os.path.join(DATADIR,category)  # create path to animals
    for img in os.listdir(path):  # iterate over each image 
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array, cmap='gray')  # graph it
        plt.show()  # display!

        break  # we just want one for now so break
    break  #...and one more!
IMG_SIZE = 100  #number of pixel

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()
training_data=[]

def create_training_data():
    for category in CATEGORIES:  # do in all animals

        path = os.path.join(DATADIR,category)  # create path to animals
        class_num = CATEGORIES.index(category)  # get the classification  (0 , 1  ,2...). as per as defined in catogery

        for img in tqdm(os.listdir(path)):  # iterate over each image per animals
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()


print(len(training_data))
import random

random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
A=X
print(len(A))
B=y
print(len(B))
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
       # rotation_range=40,
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


datagen.fit(X)
import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
                #We can always load it in to our current script, or a totally new one by doing:

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
'''from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X)
'''
#THIS MODEL NOT TOO GOOD TRY NEXT ONE GIVEN BELOW

'''#build model (output has 10 units for the 10 discrete possible classifications)
model = tf.keras.models.Sequential() # feedforward model
model.add(tf.keras.layers.Flatten()) # first layer flattens image matrix into vector
model.add(tf.keras.layers.Dense(256, activation='relu')) # first hidden layer is dense, with 128 units and uses relu for activation fxn
model.add(tf.keras.layers.Dense(312, activation='relu'))  # second hidden layer is dense, with 128 units and uses relu for activation fxn
model.add(tf.keras.layers.Dense(512, activation='relu'))  # second hidden layer is dense, with 128 units and uses relu for activation fxn
model.add(tf.keras.layers.Dense(512, activation='sigmoid'))  # second hidden layer is dense, with 128 units and uses relu for activation fxn
model.add(tf.keras.layers.Dense(312, activation='relu')) # first hidden layer is dense, with 128 units and uses relu for activation fxn
model.add(tf.keras.layers.Dense(256, activation='relu'))  # second hidden layer is dense, with 128 units and uses relu for activation fxn
model.add(tf.keras.layers.Dense(256, activation='relu'))  # second hidden layer is dense, with 128 units and uses relu for activation fxn
model.add(tf.keras.layers.Dense(64, activation='relu')) # output layer is dense, with 10 units and uses softmax for probability distribution
model.add(tf.keras.layers.Dense(10, activation='softmax')) # output layer is dense, with 10 units and uses softmax for probability distribution


# set training parameters
model.compile(optimizer='adam', # adam optimizer: good default just like relu
              loss='sparse_categorical_crossentropy', # loss=amount of error, which is what model tries to minimize
              metrics=['accuracy']) # also output accuracy during training

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)


# train model
model.fit(X, y, batch_size=32, epochs=10, validation_split=0.2) # epoch=a training cycle that iterates through all the training data once
'''

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

#import pickle

#pickle_in = open("X.pickle","rb")
#X = pickle.load(pickle_in)

#pickle_in = open("y.pickle","rb")
#y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(10))
model.add(Activation('sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y = np.array(y)

history = model.fit(X, y, batch_size=3, epochs=40, validation_split=0.3)

#model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)
from tensorflow.keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
NAME = "10animals-CNN"

from tensorflow.keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))



model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )



model.fit(X, y,
          batch_size=32,
          epochs=10,
          validation_split=0.3,
          callbacks=[tensorboard])
def predict(i): 
   
    predictions = model.predict(x_test) #predictions will be a matrix of probability dist derived via softmax previously
    print('predicted number: {}'.format(np.argmax(predictions[i]))) #pick a test choice with largest probability
    plt.imshow(x_test[i],cmap=plt.cm.binary)
    plt.show()
    
    
a=int(input('enter any number:-'))
predict(a)    