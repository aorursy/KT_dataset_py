# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../"))



# Any results you write to the current directory are saved as output.
import pandas as pd



train_set = pd.read_csv('../input/train.csv')

test_set = pd.read_csv('../input/test.csv')

test_length = len(test_set)

train_set = pd.get_dummies(train_set,columns = ['diagnosis'])
train_image_path = "../input/train_images/"

test_image_path = "../input/test_images/"
display(train_set.head())
from skimage.io import imread

import matplotlib.pyplot as plt

path = train_image_path+train_set.iloc[2,0]+".png"

image = imread(path,as_gray=True)

img = imread(path)



plt.imshow(image,cmap='gray')



plt.imshow(img)
def get_train_input(image):

    path = train_image_path+image+".png"

    img = imread(path,as_gray=True)

    return img
def get_test_input(image):

    path = test_image_path+image+".png"

    img = imread(path,as_gray=True)

    return img
def get_output(image):

    df = train_set.loc[train_set['id_code'] == image]

    df = df.iloc[:,1:]

    return df.values[0]
from skimage.transform import resize,rotate

import cv2

def preprocess_input(image):

    kernel = np.ones((5,5),np.uint8)

    erosion = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)

    erosion = resize(erosion,(100,100),anti_aliasing=False)

    erosion = rotate(erosion,np.random.randint(0,45))

    return resize(image,(100,100),anti_aliasing=False)

    


def train_image_data_generator(train_set,batch_size):

    while True:

        # Select files (paths/indices) for the batch

        batch_paths = np.random.choice(a = train_set.iloc[:,0], size = batch_size)

        batch_input = []

        batch_output = [] 

          

          # Read in each input, perform preprocessing and get labels

        for input_path in batch_paths:

            input = get_train_input(input_path)

            output = get_output(input_path)

            input = preprocess_input(image=input)

            input = np.reshape(input,(100,100,1))

            batch_input += [ input ]

            batch_output += [ output ]

          # Return a tuple of (input,output) to feed the network

        batch_x = np.array( batch_input )

        batch_y = np.array( batch_output )

        yield( batch_x, batch_y ) 
def test_image_data_generator(batch,batch_size):

        # Select files (paths/indices) for the batch

    for chunks in batch:

        batch_paths = chunks

        batch_input = []

            # Read in each input, perform preprocessing and get labels

        input = get_test_input(batch_paths[0])

        input = resize(image,(100,100),anti_aliasing=False)

        input = np.reshape(input,(100,100,1))

        batch_input += [ input ]

        # Return a tuple of (input,output) to feed the network

        batch_x = np.array( batch_input )

                

    return batch_x
from keras.layers import Conv2D,Dense,MaxPool2D,Flatten

from keras.models import Sequential

batch_size = 32

model = Sequential()

model.add(Conv2D(4,kernel_size = (3,3),padding = 'same',activation='relu',input_shape = (100,100,1)))

model.add(Conv2D(32,kernel_size = (3,3),activation='relu'))

model.add(Flatten())

model.add(Dense(64,activation = 'relu'))

model.add(Dense(32,activation = 'relu'))

model.add(Dense(5,activation = 'softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=["mae","accuracy"])

model.summary()
generator = train_image_data_generator(train_set,batch_size)



history = model.fit_generator(generator,steps_per_epoch = len(train_set)//batch_size,epochs= 15,verbose=1)
print("Loss :- "+ str(history.history['loss'][0]))

print("Mean absolute error :- "+ str(history.history['mean_absolute_error'][0]))

model.save_weights('model.h5')
import sys

test_set = pd.read_csv('../input/test.csv',chunksize= 32)

row_list = []

obj = {}

pred = pd.DataFrame(columns=['id_code','diagnosis'])

for chunks in test_set:

    data = chunks.values

    for item in data:

        image = get_test_input(item[0])

        kernel = np.ones((5,5),np.uint8)

        input = resize(image,(100,100),anti_aliasing=False)

        input = np.reshape(input,(1,100,100,1))

        out = model.predict(input)

        obj = {

            'id_code':item[0],

            'diagosis':np.argmax(out,axis=1)[0]

        }

        row_list.append(obj)

        sys.stdout.write(str(len(row_list)))

        sys.stdout.flush()

final_pred = pd.DataFrame(row_list)
final_pred.to_csv('submission.csv', index = False, header = True, sep = ',', encoding = 'utf-8')