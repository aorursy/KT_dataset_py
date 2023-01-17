# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2

import random
PATHDIR = "../input/cat-and-dog/"

categories = ["cats","dogs"]



IMG_SIZE = 100





def create_training_data():

    training_data = []

    

    for c in categories:

        for img_name in os.listdir(os.path.join(PATHDIR,"training_set/training_set",c)):

            img_path = os.path.join(PATHDIR,"training_set/training_set",c,img_name)

            img_arr = cv2.imread(img_path,0)

            try:

                resized = cv2.resize(img_arr, (IMG_SIZE,IMG_SIZE))

                norm_arr = resized/255

                training_data.append([norm_arr, categories.index(c)])

            except Exception as e:

                pass

    random.shuffle(training_data)

    return training_data



training_data = create_training_data()
X = []

y = []



for f,l in training_data:

    X.append(f)

    y.append(l)

    

X = np.array(X).reshape(-1, IMG_SIZE,IMG_SIZE,1)

y = np.array(y)
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
model = Sequential()



#First Layer

model.add(Conv2D(64, (3,3), input_shape = (IMG_SIZE,IMG_SIZE,1)))

model.add(MaxPooling2D(pool_size= (2,2)))

model.add(Flatten())



#Second Layer

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.2))



#Output Layer

model.add(Dense(10, activation = "softmax"))



model.summary()
model.compile(optimizer = "adam",

             loss ="sparse_categorical_crossentropy",

             metrics = ["accuracy"])



model.fit(X,y, epochs = 20)
def create_testing_data():

    testing_data = []

    

    for c in categories:

        for img_name in os.listdir(os.path.join(PATHDIR,"training_set/training_set",c)):

            img_path = os.path.join(PATHDIR,"training_set/training_set",c,img_name)

            img_arr = cv2.imread(img_path,0)

            try:

                resized = cv2.resize(img_arr, (IMG_SIZE,IMG_SIZE))

                norm_arr = resized/255

                testing_data.append([norm_arr, categories.index(c)])

            except Exception as e:

                pass

    random.shuffle(testing_data)

    return testing_data



test_data = create_testing_data()
X_test = []

y_test = []



for f,l in test_data:

    X_test.append(f)

    y_test.append(l)

    

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE,1)

y_test = np.array(y_test)
model.evaluate(X_test, y_test)
import matplotlib.pyplot as plt
img_path = "../input/cat-dataset/CAT_00/00000001_005.jpg"

img_arr = cv2.imread(img_path,0)

img_arr = cv2.resize(img_arr, (IMG_SIZE,IMG_SIZE))

norm_arr = img_arr/255

pred_img = (norm_arr)

    

    

pred_img = pred_img.reshape(100,100,1)
test_img = pred_img

test_img = test_img.reshape(1,IMG_SIZE,IMG_SIZE,1)

predict_ = model.predict(test_img)[0]



for i in range(len(predict_)):

    if(predict_[i] == max(predict_)):

        if(i == 0):

            print("This is a CAT!")

        else:

            print("This is a DOG!")

        

plt.imshow(test_img.reshape(IMG_SIZE,IMG_SIZE), cmap='gray')