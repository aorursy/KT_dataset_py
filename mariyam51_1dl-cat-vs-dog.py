# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!unzip /kaggle/input/dogs-vs-cats/test1.zip -d test

!unzip /kaggle/input/dogs-vs-cats/train.zip -d train

filenames = os.listdir("../working/train/train")

CATEGORIES = ["Dog","Cat"]
main_dir = "../working/"

train_dir = "train/train"

path = os.path.join(main_dir,train_dir)
import cv2

import matplotlib.pyplot as plt

for p in os.listdir(path):

    category = p.split(".")[0]

    img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)

    new_img_array = cv2.resize(img_array, dsize=(80, 80))

    plt.imshow(new_img_array,cmap="gray")

    break
X = []

y = []

convert = lambda category : int(category == 'dog')

def create_test_data(path):

    for p in os.listdir(path):

        category = p.split(".")[0]

        category = convert(category)

        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)

        new_img_array = cv2.resize(img_array, dsize=(80, 80))

        X.append(new_img_array)

        y.append(category)

    

    
create_test_data(path)

X = np.array(X).reshape(-1, 80,80,1)

y = np.array(y)

X.shape

y.shape

X =X/255
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
model = Sequential()
model = Sequential()



model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(256, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors



model.add(Dense(64))



model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
train_dir = "test/test1"

path =os.path.join(main_dir,train_dir)
X_test = []

id_line = []
def create_test1_data(path):

    for p in os.listdir(path):

        id_line.append(p.split('.')[0])

        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)

        new_img_array = cv2.resize(img_array,dsize=(80,80))

        X_test.append(new_img_array)

create_test1_data(path)
X_test = np.array(X_test).reshape(-1, 80,80,1)

X_test = X_test/255
predictions = model.predict(X_test)

predictions[3]
predicted_val = [int(round(p[0]) )for p in predictions]
submission_df = pd.DataFrame({'id' : id_line ,'label':predicted_val})
submission_df.to_csv("submission1.csv", index=False)