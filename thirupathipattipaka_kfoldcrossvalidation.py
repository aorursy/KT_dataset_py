# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#images are stored in the input
import pandas as pd
import numpy as np
labels = pd.read_csv('../input/trainLabels.csv')
#importing images
import os
from PIL import Image
import random
import matplotlib.pyplot as plt
import time

height,width,channels = 224,224,3
images_dir = os.listdir('../input')
images_dir.remove('trainLabels.csv')
img_matrix = []
img_label = []

for file in images_dir:
    base = os.path.basename("../input/" + file)
    fileName = os.path.splitext(base)[0]
    img_label.append(labels.loc[labels.image==fileName, 'level'].values[0])
    im = Image.open("../input/" + file)   
    img = im.resize((height,width))
    rgb = img.convert('RGB')
    img_matrix.append(np.array(rgb).flatten())
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

img_matrix = np.asarray(img_matrix)
img_label = np.asarray(img_label)

data,label = shuffle(img_matrix,img_label,random_state=2)
train_data = [data,label]
img = img_matrix[200].reshape(height,width,channels)
plt.imshow(img)
(X,Y) = (train_data[0],train_data[1])
#categorical
classes = 5
epochs = 5
batchsize=128

from keras.models import Sequential
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold


X.shape
Y.shape
X = X.reshape(X.shape[0],height,width,channels)
X = X.astype('float32')
X /= 255
# 10 fold cross validation
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=7)
cvscores = []
for train,test in kfold.split(X,Y):
    # create the cnn model
    model = Sequential()
    #add model layers
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(height,width,channels), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64,(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64,(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64,(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    #optimizer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    #model fit
    Y_train = np_utils.to_categorical(Y[train],classes)
    model.fit(X[train], Y_train,epochs=epochs,verbose=0)
    
    #evaluate the mode
    Y_test = np_utils.to_categorical(Y[test],classes)
    scores = model.evaluate(X[test],Y_test,verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1]*100)

print("%.2f%%" % np.mean(cvscores))
# 10 fold cross validation
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=7)
cvscores = []
for train,test in kfold.split(X,Y):
    # create the cnn model
    model = Sequential()
    #add model layers
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(height,width,channels), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64,(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64,(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64,(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    #optimizer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    Y_train = to_categorical(Y[train],classes)
    #model fit
    print(X[train].shape,Y_train.shape)
from keras.utils import to_categorical
to_categorical(Y[1:10],5).shape
