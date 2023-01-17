import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding,Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img

import cv2
from pathlib import Path
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
# Define path to the data directory
data_dir = Path('../input/food-nonfood-dataset/Data')

# Path to train directory
train_dir = data_dir / 'train'

# Path to test directory
test_dir = data_dir / 'test'
img_size = 240
img_shape = (img_size,img_size,3)

def make_data(some_dir, img_size = 240):
    '''
    To make X,y for Train and Test
    '''
    
    # set paths
    food_dir = some_dir / 'food'
    nonfood_dir = some_dir / 'nonfood'

    # get all jpg
    food_imgs = food_dir.glob('*.jpg')
    nonfood_imgs = nonfood_dir.glob('*.jpg')

    X = []
    y = []

    for img in food_imgs:
        temp_img = cv2.imread(str(img)) # to img
        temp_img = cv2.resize(temp_img,(img_size,img_size)) # all different sizes, but let's try
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        temp_img = temp_img.astype(np.float32)/255.
        X.append(temp_img)
        y.append(1)


    for img in nonfood_imgs:
        temp_img = cv2.imread(str(img)) # to img
        temp_img = cv2.resize(temp_img,(img_size,img_size))
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        temp_img = temp_img.astype(np.float32)/255.
        X.append(temp_img)
        y.append(0)
        
    return (np.array(X),np.array(y))
# may take a while...
(X_train, y_train) = make_data(train_dir)
(X_test, y_test) = make_data(test_dir)
print("Total images are: " + str(y_train.shape[0]+y_test.shape[0]))
print("Food images are: "+ str(sum(y_train)+sum(y_test)))
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), input_shape=img_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dropout(0.9))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.9))
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=15,validation_split=0.1,batch_size=256)

model.evaluate(X_test,y_test)