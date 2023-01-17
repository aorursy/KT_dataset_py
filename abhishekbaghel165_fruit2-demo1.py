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
import cv2



y = []

X = []

training_dir = '../input/fruits-360_dataset/fruits-360/Training'



for root,dirs,files in os.walk(training_dir):

    directory = dirs

    for file in files:

        if file.endswith('.jpg'):

            im = cv2.imread(os.path.join(root,file))

            X.append(im)

            y.append(root.replace('../input/fruits-360_dataset/fruits-360/Training/',''))
from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split



encoder = LabelEncoder().fit(y)

y_train = encoder.transform(y)



X_train = np.stack(X,axis = 0)

y_train = to_categorical(y_train)

X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size = 0.2)



X_train = X_train.astype('float32')

X_val = X_val.astype('float32')

X_train /= 255

X_val /= 255
print(X_train.shape,y_train.shape)

print(X_val.shape,y_val.shape)
import matplotlib.pyplot as plt

%matplotlib inline



plt.imshow(X_train[1000,:,:,:].reshape((100,100,3)) )
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout

from keras.models import Sequential



model = Sequential()

model.add(Conv2D(4,kernel_size = 5, padding = 'same', activation = 'relu',input_shape = (100,100,3), name = 'conv0'))

model.add(MaxPooling2D())  

model.add(Conv2D(8,kernel_size = 5, padding = 'same', activation = 'relu', name = 'conv1'))

model.add(MaxPooling2D()) 

model.add(Conv2D(16,kernel_size = 5, padding = 'same', activation = 'relu', name = 'conv2'))

model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(103,activation = 'softmax',name = 'fc1'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
hist = model.fit(X_train,y_train, epochs = 20, verbose = 1, validation_data = (X_val,y_val))
y = []

X = []

test_dir = '../input/fruits-360_dataset/fruits-360/Test'



for root,dirs,files in os.walk(test_dir):

    directory = dirs

    for file in files:

        if file.endswith('.jpg'):

            im = cv2.imread(os.path.join(root,file))

            X.append(im)

            y.append(root.replace('../input/fruits-360_dataset/fruits-360/Test/',''))

            

y_train = encoder.transform(y)

y_train = to_categorical(y_train)



X_train = np.stack(X, axis = 0)

X_train = X_train.astype('float32')

X_train /= 255



print(X_train.shape,y_train.shape)
acc = model.evaluate(X_train,y_train)

print("Accuracy is :" + str(acc[1]))

print("Loss is :" + str(acc[0]))