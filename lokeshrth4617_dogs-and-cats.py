# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os



from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D,Dropout,Dense,Activation,Flatten,MaxPooling2D

import pickle

import matplotlib.pyplot as plt

import cv2

DATADIR = '../input/cat-and-dog/training_set/training_set'

categories = ['dogs','cats']

for category in categories:

    path = os.path.join(DATADIR,category) # puts to cats and dog dir

    for img in os.listdir(path):

        img_arry = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

        plt.imshow(img_arry,cmap = 'gray')

        plt.show()

        break

    break







# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print(img_arry.shape)
new_array = cv2.resize(img_arry,(60,60))

#print(new_array)

plt.imshow(new_array,cmap ='gray')

plt.show()
training_data = []

IMG_SIZE = 60

def create_training_data():

    for category in categories:

        path = os.path.join(DATADIR,category) # puts to cats and dog dir

        class_num = categories.index(category)

        for img in os.listdir(path):

            try:

                img_arry = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

                new_array = cv2.resize(img_arry,(IMG_SIZE,IMG_SIZE))

                training_data.append([new_array,class_num])

            except Exception as e:  

                    pass

               

            

create_training_data()
print(len(training_data))
import random

random.shuffle(training_data)
for sample in training_data:

    print(sample[1])
X = []

y = []
IMG_SIZE = 60

for features,label in training_data:

    X.append(features)

    y.append(label)

    

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
## this is for using this dataset sometime else

## names are given below



import pickle

pickle_out = open("Dogs_and_Cats_X.pickle","wb")

pickle.dump(X ,pickle_out)

pickle_out.close()





pickle_out= open("Dogs_and_Cats_y","wb")

pickle.dump(y ,pickle_out)

pickle_out.close()
X = X/255.0
y = np.array(y)
model = Sequential()

model.add(Conv2D(80,(3,3), input_shape =X.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(80,(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(80))

model.add(Activation('sigmoid'))

model.add(Dropout(0.2))





model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss = 'binary_crossentropy',

             optimizer = 'adam',

             metrics = ['accuracy'])



model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)
pred = model.predict(X[0][0][0][0])

plt.imshow(pred,cmap ='gray')

plt.show()