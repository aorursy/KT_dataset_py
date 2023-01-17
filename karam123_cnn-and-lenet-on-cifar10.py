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
!pip install py7zr
import pandas as pd
from py7zr import unpack_7zarchive
import shutil
from PIL import Image as PImage
from matplotlib import pyplot as plt
import keras
import tensorflow as tf
check = pd.read_csv('/kaggle/input/cifar-10/trainLabels.csv')
'''shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)
shutil.unpack_archive('/kaggle/input/cifar-10/train.7z', '/kaggle/working/train)'''
'''shutil.rmtree('/kaggle/working/train')'''
'''os.listdir('/kaggle/working/train/train')'''
'''def loadImages(path):
    # return array of images

    imagesList = os.listdir(path)
    loadedImages = []
    for image in imagesList:
        img = PImage.open(path + image)
        loadedImages.append(img)

    return loadedImages
'''
'''path = '/kaggle/working/train/train/'
train_images = loadImages(path)'''
#train_images[0]
#plt.imshow(train_images[0])
#plt.show()
#train_images[0].size
from keras.datasets import cifar10

data = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = data.load_data()
x_train.shape
x_train[0]
plt.imshow(x_train[0])
plt.show()
from collections import Counter
b=[]
for i in range(len(y_train)):
    y_train[i][0]
    b.extend([y_train[i][0]])
a = Counter(b)
a
new = sorted(a.items())
new = dict(new)
labels = new.keys()
labels
values = new.values()
values
index = np.arange(len(labels))
plt.bar(index,values)
plt.xlabel('class')
plt.ylabel('occurance')
plt.xticks(index,labels)
plt.title('occurance of different classes')
plt.show()
b=[]
for i in range(len(y_test)):
    y_test[i][0]
    b.extend([y_test[i][0]])
a = Counter(b)
a
new = sorted(a.items())
new = dict(new)
new
labels = new.keys()
labels
values = new.values()
values
plt.bar(index,values)
plt.xlabel('class')
plt.ylabel('occurance')
plt.xticks(index,labels)
plt.title('occurance of different classes')
plt.show()
a =x_train[0]
np.amax(a)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
a =x_train[0]
np.amax(a)
num_classes = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)
y_train[0]
for_tsne = []
for i in range(1000):
    a = x_train[i]
    for_tsne.append(a.flatten())
for_tsne = np.array(for_tsne)
for_tsne.shape
from sklearn.manifold import TSNE

# Picking the top 1000 points as TSNE takes a lot of time for 15K points
data_1000 = for_tsne
labels_1000 = y_train[0:1000]

model = TSNE(n_components=2, random_state=0)
# configuring the parameteres
# the number of components = 2
# default perplexity = 30
# default learning rate = 200
# default Maximum number of iterations for the optimization = 1000

tsne_data = model.fit_transform(data_1000)

# creating a new data frame which help us in ploting the result data
tsne_data = np.hstack((tsne_data, labels_1000))
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

import seaborn as sn
# Ploting the result of tsne
sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
x_train[0].shape
input_shape = (32,32,3)
batch_size =100
epochs =10
model  = Sequential()

model.add(Conv2D(128,(3,3),activation = 'relu',input_shape = input_shape))
model.add((MaxPooling2D(pool_size = (2,2))))

model.add(Conv2D(256,(3,3),activation = 'relu'))
model.add((MaxPooling2D(pool_size = (2,2))))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))


model.summary()
model.compile(loss= 'categorical_crossentropy',optimizer = 'adam' , metrics = ['accuracy'])
history=model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
%matplotlib notebook
%matplotlib inline

import time
# https://gist.github.com/greydanus/f6eee59eaf1d90fcb3b534a25362cea4
# https://stackoverflow.com/a/14434334
# this function is used to update the plots for each epoch and error
def plt_dynamic(x, vy, ty, ax, colors=['b']):
    ax.plot(x, vy, 'b', label="Validation Loss")
    ax.plot(x, ty, 'r', label="Train Loss")
    #plt.legend()
    plt.grid()
    plt.show()
    fig.canvas.draw()
fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,epochs+1))

# print(history.history.keys())
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
# history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# we will get val_loss and val_acc only when you pass the paramter validation_data
# val_loss : validation loss
# val_acc : validation accuracy

# loss : training loss
# acc : train accuracy
# for each key in histrory.histrory we will have a list of length equal to number of epochs

vy = history.history['val_accuracy']
ty = history.history['accuracy']
plt_dynamic(x, vy, ty, ax)
model  = Sequential()

model.add(Conv2D(6,(5,5),activation = 'relu',input_shape = input_shape))
model.add((MaxPooling2D(pool_size = (2,2))))

model.add(Conv2D(16,(5,5),activation = 'relu'))
model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Flatten())

model.add(Dense(120,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(84,activation = 'relu'))

model.add(Dense(10,activation = 'softmax'))
model.summary()
model.compile(loss= 'categorical_crossentropy',optimizer = 'adam' , metrics = ['accuracy'])
history=model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,epochs+1))

# print(history.history.keys())
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
# history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# we will get val_loss and val_acc only when you pass the paramter validation_data
# val_loss : validation loss
# val_acc : validation accuracy

# loss : training loss
# acc : train accuracy
# for each key in histrory.histrory we will have a list of length equal to number of epochs

vy = history.history['val_loss']
ty = history.history['loss']
plt_dynamic(x, vy, ty, ax)