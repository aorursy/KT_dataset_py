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

%matplotlib inline 

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA

import functools
from PIL import ImageOps, Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf


import functools
from PIL import ImageOps, Image
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

import cv2

#To see our directory
import os
import random
import gc

import PIL

import pandas as pd

my_data = pd.read_csv('../input/football-player-number-13/train_solutions.csv')
print(my_data)
ids = list(my_data['Id'])
#print(ids)
predicted = list(my_data['Predicted'])
#print(predicted)

my_data = pd.read_csv('../input/football-player-number-13/train_solutions.csv')
print(my_data)
training_data=[]
def create_training_data():
    for i in range(len(ids)):
        image = Image.open('../input/football-player-number-13/images/' +str(ids[i])[:32] + '_' + str(ids[i])[33:35] + ".jpg")
        img = image.resize((250,250))
        img_array = img_to_array(img)
        training_data.append(img_array)
        


create_training_data()
X = training_data
y = predicted
#Convert list to numpy array
X = np.array(X)
y = np.array(y)
y = y.astype(int)
print(y)
print("Shape of train images is:", X.shape)
print("Shape of labels is:", y.shape)
import seaborn as sns


#Convert list to numpy array
X = np.array(X)
y = np.array(y)

#Lets plot the label to be sure we just have two class
sns.countplot(y)
plt.title('Labels for 0 and 1')



teszt_adata = pd.read_csv('../input/football-player-number-13/sampleSubmissionAllZeros.csv')
print(teszt_adata)
teszt_ids = list(teszt_adata['Id'])
#print(ids)
teszt_data = []

def create_test_data():
    for i in range(len(teszt_ids)):
        image = Image.open('../input/football-player-number-13/images/' +str(ids[i])[:32] + '_' + str(ids[i])[33:35] + ".jpg")
        img = image.resize((250,250))
        img_array = img_to_array(img)
        teszt_data.append(img_array)

create_test_data()
x_teszt = teszt_data
x_teszt = np.array(x_teszt)
print("Shape of train images is:", x_teszt.shape)


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

print("Shape of train images is:", X_train.shape)
print("Shape of validation images is:", X_val.shape)
print("Shape of labels is:", y_train.shape)
print("Shape of labels is:", y_val.shape)
gc.collect()
#get the length of the train and validation data
ntrain = len(X_train)
nval = len(X_val)

#We will use a batch size of 32. Note: batch size should be a factor of 2.***4,8,16,32,64...***
batch_size = 32
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)

model = models.Sequential()


model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(250, 250, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  #Dropout for regularization
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  #Sigmoid function at the end because we have just two classes
model.summary()
from keras.optimizers import Adam
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
#Lets create the augmentation configuration
#This helps prevent overfitting, since we are using a small dataset
train_datagen = ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1
                                    #rotation_range=40,
                                    #width_shift_range=0.2,
                                    #height_shift_range=0.2,
                                    #shear_range=0.2,
                                    #zoom_range=0.2,
                                    vertical_flip=True,)

val_datagen = ImageDataGenerator(rescale=1./255)  #We do not augment validation data. we only perform rescale

#Create the image generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
#class_weight={0: 1., 1: 13.5}
#The training part
#We train for 64 epochs with about 100 steps per epoch
history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              class_weight={0: 1, 1: 13.5965},
                              epochs=120,
                              validation_data=val_generator,
                              validation_steps=nval // batch_size
                              )
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()
#Now lets predict on the first 10 Images of the test set
print("Shape of train images is:", x_teszt.shape)
x_pred = np.array(x_teszt)
test_datagen = ImageDataGenerator(rescale=1./255)
teszt_generator = test_datagen.flow(x_pred, batch_size=1)
predictions = model.predict(teszt_generator)
print(predictions)
print("Shape of prediction is:", predictions.shape)
labels = []
for i in range(len(predictions)):
    if predictions[i] > 0.95:
        labels.append(True)
    else:
        labels.append(False)

print(labels)
teszt_adata = pd.read_csv('../input/football-player-number-13/sampleSubmissionAllZeros.csv')
print(teszt_adata)
teszt_adata['Predicted'] = labels

print(teszt_adata)
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(teszt_adata)
