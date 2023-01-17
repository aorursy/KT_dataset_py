

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import cv2







folder=("/kaggle/input/brain-mri-images-for-brain-tumor-detection")

data=[]

data_set=[]

data_label=[]

for files in os.listdir(folder):

    path0=os.path.join(folder,files)

    

    if (path0.find("yes")!= -1): 

        for path2 in os.listdir(path0):

            try:

                img_file=cv2.imread(os.path.join(path0,path2))

           

                img_file1=cv2.resize(img_file,(60,60))

                data.append([img_file1,1])

               

            except Exception as es:

                pass

    elif (path0.find("no")!= -1):

        for path3 in os.listdir(path0):

            try:

                img_file=cv2.imread(os.path.join(path0,path3))

    

                img_file1=cv2.resize(img_file,(60,60))

                data.append([img_file1,0])

               

            except Exception as es:

                pass

    else:

      

         for path4 in os.listdir(path0):

            path3=os.path.join(path0,path4)

            if path3.find("yes")!= -1:# or path2.find("Yes"):

                for img in os.listdir(path3):

                    try:    

                        img_file=cv2.imread(os.path.join(path3,img))

                           

                        

                        img_file1=cv2.resize(img_file,(60,60))

                        data.append([img_file1,1])

                        

                    except Exception as es:

                        pass

            else:

                for img in os.listdir(path3):

                    try:

                        img_file=cv2.imread(os.path.join(path3,img))

                    

                        img_file1=cv2.resize(img_file,(60,60))

                        data.append([img_file1,0])

                        

                    except Exception as es:

                        pass



        







data=np.array(data)

np.random.shuffle(data)

for features,label in data:

    data_set.append(features)

    data_label.append(label)

    
from keras.models import Sequential

from keras.layers import Flatten

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Dense

data_set=np.array(data_set)

data_label=np.array(data_label)

print(data_set.shape)

print(data_label.shape)
from sklearn.model_selection import train_test_split

X_train,X_test=train_test_split(data_set,test_size=0.2)

Y_train,Y_test=train_test_split(data_label,test_size=0.2)



X_train.reshape(-1,60,60,1)

X_test.reshape(-1,60,60,1)

print(X_train.shape)

print(X_train.shape[1:])

#print(Y_train[0:25])
clf=Sequential()

clf.add(Convolution2D(64,(3,3),input_shape=(60,60,3),data_format='channels_last',activation='relu'))

clf.add(MaxPooling2D(pool_size=(2,2)))

clf.add(Convolution2D(64,3,3,activation='relu'))

clf.add(Convolution2D(64,3,3,activation='relu'))

clf.add(MaxPooling2D(pool_size=(2,2)))

clf.add(Convolution2D(64,3,3,activation='relu'))

clf.add(Convolution2D(64,3,3,activation='relu'))

clf.add(MaxPooling2D(pool_size=(2,2)))

clf.add(Flatten())

clf.add(Dense(48,activation='relu'))

clf.add(Dense(48,activation='relu'))

clf.add(Dense(1,activation='sigmoid'))

clf.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

clf.fit(X_train,Y_train,batch_size=20,epochs=200)



predicted=clf.predict(X_test)

for i in range(len(predicted)):

    if predicted[i]>0.35:

        predicted[i]=1

    else:

        predicted[i]=0

from sklearn.metrics import confusion_matrix

mat=confusion_matrix(Y_test,predicted)

print(mat)