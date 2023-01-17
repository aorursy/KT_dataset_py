# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
Directory = "/kaggle/input/dog-vs-cat-fastai/dogscats/train"
Categories =['cats','dogs']
img_size = 120
data=[]

count=0
for category in Categories:
    folder =os.path.join(Directory, category)
    label = Categories.index(category)
    for img in os.listdir(folder):
        img_pth = os.path.join(folder,img)
        print(img_pth)
        img_arr=cv2.imread(img_pth)
        img_arr = cv2.resize(img_arr,(img_size, img_size)) #resizing the image
        plt.imshow(img_arr)
        data.append([img_arr, label])
        count=1+count
        if(count==5000):
            break
    
random.shuffle(data)
X =[]
y =[]

for feature, label in data:
    X.append(feature)
    y.append(label)
    
X = np.array(X)
y = np.array(y)
X =X/255
X.shape 
X[1:].shape
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = Sequential()

model.add(Dense(128, input_shape=X.shape[1:],activation='relu'))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))          
          

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))         

model.add(Flatten())
model.add(Dense(2, activation="softmax"))
model.compile(optimizer ='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X, y, epochs=5,validation_split= 0.1)
img_arr=cv2.imread('/kaggle/input/dog-vs-cat-fastai/dogscats/valid/cats/cat.7883.jpg')
img_arr_cat = cv2.resize(img_arr,(img_size, img_size))
img_arr_cat=np.array(img_arr_cat)
img_arr_cat.shape
prediction=model.predict(img)
print(prediction)