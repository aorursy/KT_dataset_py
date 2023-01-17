# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input/train"))
#cat.11679.jpg
#dog.2578.jpg
# Any results you write to the current directory are saved as output.
#Get label from image name, for Cat, return 0 nd for dog, return 1
def get_label(img_name):
    idx = img_name.find('.')
    label = img_name[:idx]
    #return label
    if label == 'cat':
        return 0
    else:
        return 1   
from skimage.io import imread
import cv2 as cv
#This function reads images from path given and resize it
def get_img_files(img_path,name):
    file_name = img_path+'/'+str(name)
    img = imread(file_name)
    img = cv.resize(img,(img_cols,img_rows),3)
    return img        
img_path = '../input/train'
#This function is to create custom datagenerator
def get_data(batch_size = 64):
    img_files = os.listdir("../input/train")
    while True:
        # get files as per batch size
        files = np.random.choice(a = img_files, size = batch_size)
        ite_input = []
        ite_output = []           
        #Read each input image and get labels
        for file in files:
            dog_cat = get_img_files(img_path,file)
            label = get_label(file)            
            ite_input.append(dog_cat)
            ite_output.append(label)
            
        # Return a tuple of (input,output) to feed the network
        x = np.array( ite_input)
        y = np.array( ite_output )
        y=keras.utils.to_categorical(y, num_classes=num_classes)
        yield(x, y)        

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation,Dense, Flatten, Conv2D, Dropout,MaxPooling2D
from tensorflow.python import keras
from tensorflow.python.keras.optimizers import RMSprop
opti = RMSprop(lr=1e-4)
loss_function = 'binary_crossentropy'
act_last = 'sigmoid'
#act_last = 'softmax'
act = 'relu'

num_classes = 2
img_rows = 64
img_cols = 64
img_depth = 3

#Create a model
def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),padding='same',
                      activation=act,
                      input_shape=(img_cols, img_rows, img_depth)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same',activation=act))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same',activation=act))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same',activation=act))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same',activation=act))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same',activation=act))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same',activation=act))
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same',activation=act))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())    
    model.add(Dense(256, activation=act))    
    model.add(Dropout(0.5))
    model.add(Dense(256, activation=act))    
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=act_last))

    model.compile(loss=loss_function,
                  optimizer=opti,
                  metrics=['accuracy'])    
    return model
from keras.applications.resnet50 import preprocess_input
def _load_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
# #get new model and train with new data
my_new_model = get_model()
my_new_model.fit_generator(
        get_data(),
        steps_per_epoch=500,
        epochs=6,
        #validation_data=validation_generator,
        #validation_steps=1
        )

#read test images and capture predictions
test_dir = '../input/test/'
test_img_files = os.listdir(test_dir)
test_files = np.random.choice(a = test_img_files, size = 20)
ite_input_test = []
#Read each input image and get labels
for test_file in test_files:
    test_dog_cat = get_img_files(test_dir,test_file)
    ite_input_test.append(test_dog_cat)

# Return a tuple of (input,output) to feed the network
test_x = np.array( ite_input_test)

predictions = my_new_model.predict(test_x, verbose=0)
print(predictions)
#check for first 20 predictions
import matplotlib.pyplot as plt
for i in range(0,20):
    if predictions[i, 0] >= 0.5: 
        print('I am {:.2%} sure this is a cat'.format(predictions[i][0]))
    else: 
        print('I am {:.2%} sure this is a Dog'.format(1-predictions[i][0]))
    
    plt.imshow(ite_input_test[i])
    plt.show()    
    