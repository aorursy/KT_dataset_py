# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
print('strat kar le ab')
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import keras
from tqdm import tqdm
import tensorflow as tf
# from IPython.display import Image, display
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, Dropout,MaxPooling2D
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from time import time
import os
from PIL import Image
print(os.listdir("../input"))

import matplotlib.pyplot as plt
DATADIR="../input/waste-classification-data/DATASET/TRAIN"
DATADIR1="../input/waste-classification-data/DATASET/TEST"

CATEGORIES=['O','R']
# for cats in CATEGORIES:
#     path= os.path.join(DATADIR,cats)
#     for img in os.listdir(path):
#         imge=Image.open(os.path.join(path,img))
#         imge=imge.resize((150,150))
        
#         img_array= np.array(imge)
#         print(img_array.shape)
#         plt.imshow(img_array)
#         plt.show()
#         break
      
   
test=[]
train = []
image_size=128

def create_data(DIR,arr):
    i=0
    for category in CATEGORIES: 

        path = os.path.join(DIR,category) 
        class_num = np.array([CATEGORIES.index(category)]  )
        print(class_num)
        
        print(i)
        for img in tqdm(os.listdir(path)):  
            try:
                imge=Image.open(os.path.join(path,img))
                imge=imge.resize((image_size,image_size))
                img_array= np.array(imge)
                i=+1
                
                if img_array.shape==(image_size,image_size,3):
                    arr.append([img_array, class_num])
                    
            except Exception as e:
                print(e)
training=create_data(DATADIR,train)
testing=create_data(DATADIR1,test)
# print(train.shape)
import random
random.shuffle(train)
random.shuffle(test)
type(train[2])
f,t = train[0]
# print("f",f,"t",t)
print(f[0])
# x_train=np.array([])
# y_train=np.array([])
# x_test=np.array([])
# y_test=np.array([])
# for feature,label in train:
#     x_train=np.append(x_train,feature)
#     y_train=np.append(y_train,label)
# y_train = keras.utils.to_categorical(y_train, 2)

# for feature,label in test:
#     x_test=np.append(x_test,feature)
#     y_test=np.append(y_test,label)
# y_test = keras.utils.to_categorical(y_test, 2)
x_train=[]
y_train=[]
x_test=[]
y_test=[]
for feature,label in train:
    x_train.append(feature)
    y_train.append(label)
    

y_train = keras.utils.to_categorical(y_train, 2)

for feature,label in test:
    x_test.append(feature)
    y_test.append(label)
    
y_test = keras.utils.to_categorical(y_test, 2)
x_train=np.array(x_train)
x_test=np.array(x_test)

x_test=x_test/255
x_train=x_train/255
print(x_train)
y_train
import pickle

# pickle_out = open("x_train.pickle","wb")
# pickle.dump(x_train, pickle_out)
# pickle_out.close()

# pickle_out = open("y_train.pickle","wb")
# pickle.dump(y_train, pickle_out)
# pickle_out.close()

# pickle_out = open("x_test.pickle","wb")
# pickle.dump(x_test, pickle_out)
# pickle_out.close()

# pickle_out = open("y_test.pickle","wb")
# pickle.dump(y_test, pickle_out)
# pickle_out.close()
x_test=0
print(x_test)
pickle_in = open("x_test.pickle","rb")
x_test = pickle.load(pickle_in)

print(x_test)
model=Sequential()
model.add(Conv2D(40,kernel_size=(3,3),strides=(2,2),activation='relu', input_shape=(150,150,3)))
model.add(Conv2D(100,kernel_size=(4,4),activation='relu'))

model.add(Conv2D(80, kernel_size=(3,3),activation='relu'))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(80,activation='relu'))
model.add(Dropout(.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy'])

# model=Sequential()
# model.add(Conv2D(filters=40,kernel_size=(3,3),padding="Same",activation="relu",input_shape=(100,100,3)))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(filters=60,kernel_size=(3,3),padding="Same",activation="relu"))
# model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(filters=100,kernel_size=(3,),padding="Same",activation="relu"))
# model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(120,activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(2,activation='softmax'))
# model.compile(loss='categorical_crossentropy',
#               optimizer='adagrad',
#               metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=178, epochs=50,steps_per_epoch=100, validation_split=0.2)
k=np.array([])
z=np.array([])

for i in range(3):
    z=np.append(k,i)
    print(z)
type(k)