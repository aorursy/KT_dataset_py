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
from keras.models import Sequential

from keras.layers import Convolution2D, MaxPooling2D,Flatten,Dense,Dropout ,BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

import os

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img



import cv2

import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split

import random
data = []

uninfected = os.listdir("../input/cell_images/cell_images/Uninfected")

parasitized = os.listdir("../input/cell_images/cell_images/Parasitized")



for i in uninfected:

    data.append(["cell_images/Uninfected/"+i,0])

for i in parasitized:

    data.append(["cell_images/Parasitized/"+i,1])

random.shuffle(data)

image = [i[0] for i in data]

label = [i[1] for i in data]

del data





X_train, X_test, Y_train, Y_test = train_test_split(image, label, test_size=0.3, random_state=123)

# X_train.shape
chanDim = -1

classifier=Sequential()

#convolution

classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))

#pooling

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(BatchNormalization(axis = chanDim))

classifier.add(Dropout(0.2))

##second CNN layer

classifier.add(Convolution2D(32,(3,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(BatchNormalization(axis = chanDim))

classifier.add(Dropout(0.2))



#flattening

classifier.add(Flatten())

#full connection

classifier.add(Dense(output_dim=128,activation='relu'))

classifier.add(BatchNormalization(axis = chanDim))

classifier.add(Dropout(0.2))

#output layer

classifier.add(Dense(output_dim=1,activation='sigmoid'))



#compile

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



# classifier.fit(X)
classifier.summary()
classifier.fit(np.array(X_train),np.array(Y_train),steps_per_epoch = len(X_train)//64 , epochs=25,validation_split = 0.2)
