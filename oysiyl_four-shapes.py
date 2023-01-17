# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input/shapes/*"))

# Any results you write to the current directory are saved as output.
import glob
import cv2
from sklearn.utils import shuffle

import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.layers.normalization import BatchNormalization
#Define a path to our data
circle_train_path = '../input/shapes/*circle'
square_train_path = '../input/shapes/*square'
star_train_path = '../input/shapes/*star'
triangle_train_path = '../input/shapes/*triangle'

cir_path = os.path.join(circle_train_path,'*g')
sq_path = os.path.join(square_train_path,'*g')
str_path = os.path.join(star_train_path,'*g')
trg_path = os.path.join(triangle_train_path,'*g')
#create train array
x_train = []
y_train = []
print("Reading Training Data")
#reading data to memory
file_1 = glob.glob(cir_path)
for f1 in file_1:
    img = cv2.imread(f1)
    x_train.append(img)
    y_train.append([1,0,0,0])
    
print("25% Complete")
    
file_2 = glob.glob(sq_path)
for f2 in file_2:
    img = cv2.imread(f2)
    x_train.append(img)
    y_train.append([0,1,0,0])
    
print("50% Complete")
    
file_3 = glob.glob(str_path)
for f3 in file_3:
    img = cv2.imread(f3)
    x_train.append(img)
    y_train.append([0,0,1,0])
    
print("75% Complete")
    
file_4 = glob.glob(trg_path)
for f4 in file_4:
    img = cv2.imread(f4)
    x_train.append(img)
    y_train.append([0,0,0,1])
    
x_train = np.array(x_train)
y_train = np.array(y_train)

x_train, y_train = shuffle(x_train,y_train)

print("Finished Reading Training Data")
#create a model
model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape = (200,200,3)))
model.add(MaxPooling2D((2,2),strides = (2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64,(2,2),activation='relu'))
model.add(MaxPooling2D((2,2),strides = (2,2)))
model.add(Conv2D(128,(2,2),activation='relu'))
model.add(MaxPooling2D((2,2),strides = (2,2)))
model.add(BatchNormalization())
model.add(Conv2D(256,(5,5),activation='relu'))
model.add(MaxPooling2D((2,2),strides = (2,2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(4096,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(4,activation='softmax'))
#model compile, fit and save
model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.SGD(lr=0.00001),
             metrics = ['accuracy'])

model.fit(x_train,y_train,batch_size = 1,epochs = 2,verbose = 1,validation_split = 0.0)
model.save('shape_recg.h5')
#del model
#load our model again
from keras.models import load_model
model = load_model('shape_recg.h5')
import cv2
import numpy as np
#test model on image
dir_path = '../input/shapes/circle/2.png'
test_img = cv2.imread(dir_path)
test_img = test_img.reshape(1,200,200,3)

Y = model.predict(test_img)[0]
val = np.argmax(Y)
if(val == 0):
    print("Circle")
elif(val == 1):
    print("Square")
elif(val == 2):
    print("Star")
else:
    print("Triangle")