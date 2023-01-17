import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import os

import cv2

import matplotlib.pyplot as plt

filenames=["no","yes"]

training_data=[]

data_set=[]

data_label=[]

execution_path=os.getcwd()

dirname='/kaggle/input/brain-mri-images-for-brain-tumor-detection/brain_tumor_dataset/'





for filename in filenames:

    path=os.path.join(dirname,filename)

    filename_index=filenames.index(filename)

    for image in os.listdir(path):

        try:

            img_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)

            img_array = cv2.resize(img_array, (128,128))

            training_data.append([img_array, filename_index])

        except Exception as e:

            print(e)

training_data=np.array(training_data)

np.random.shuffle(training_data)



for feature, label in training_data:

    data_set.append(feature)

    data_label.append(label)

data_label=np.array(data_label)

data_set=np.array(data_set)



x_train,x_test=train_test_split(data_set,test_size=0.1)

y_train,y_test=train_test_split(data_label,test_size=0.1)



x_train = np.array(x_train).reshape(-1,128,128,1)

x_test = np.array(x_test).reshape(-1,128,128,1)



y_train=np.array(y_train)

y_test=np.array(y_test)



from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Dense

from keras.layers import Flatten
classifier=Sequential()

classifier.add(Convolution2D(96,11,11, input_shape=(128,128,1),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))

classifier.add(Convolution2D(256,5,5 ,activation='relu'))

classifier.add(MaxPooling2D(pool_size=(3,3),strides=2))

classifier.add(Convolution2D(384,3,3 ,activation='relu'))

classifier.add(Convolution2D(384,3,3 ,activation='relu'))

classifier.add(Convolution2D(256,3,3,activation='relu'))

classifier.add(MaxPooling2D(pool_size=(3,3),strides=2))

classifier.add(Flatten())

classifier.add(Dense(output_dim=4096,activation='relu'))

classifier.add(Dense(output_dim=4096,activation='relu'))

classifier.add(Dense(output_dim=1,activation='sigmoid'))



classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(x_train,y_train,nb_epoch=50)

y_pred=classifier.predict(x_test)

y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_pred,y_test)

cm