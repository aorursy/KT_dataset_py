#importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from skimage import io, transform
#saving the path of input images for tarining and testing

train_path='../input/train/'

test_path='../input/test/'
#importing the class labels

y=pd.read_csv('../input/train_labels.csv')

y.head()
#checking for images



im2=io.imread(train_path+'5.jpg')

plt.imshow(im2)
#creating numpy arrays for storing images

#here we are taking 200 images for training

#& 100 images for testing

#all images have size of 64x64

X=np.empty(shape=(300,64,64,3))

y=y.iloc[:300,1].values

#saving image as numpy array

for i in range(300):

    im=io.imread(train_path+str(i+1)+'.jpg')

    X[i]=transform.resize(im,output_shape=(64,64,3))
#creating training and test set

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=(1/3),random_state=0)

y
#creating our cnn model

import keras

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout
cnn=Sequential()

cnn.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

cnn.add(MaxPooling2D(pool_size=(2,2)))

cnn.add(Convolution2D(64,3,3,activation='relu'))

cnn.add(MaxPooling2D(pool_size=(2,2)))

cnn.add(Convolution2D(128,3,3,activation='relu'))

cnn.add(MaxPooling2D(pool_size=(2,2)))







cnn.add(Flatten())

cnn.add(Dense(output_dim=100,activation='relu'))

cnn.add(Dropout(p=0.3))

cnn.add(Dense(output_dim=100,activation='relu'))

cnn.add(Dropout(p=0.3))



cnn.add(Dense(output_dim=1,activation='sigmoid'))

cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



#Training our CNN model

cnn.fit(X_train,y_train,batch_size=10,epochs=20)
y_predict=cnn.predict_classes(X_test)
y_predict
#checking the accuracy

from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_test,y_predict)

cm
print(accuracy_score(y_test,y_predict))
X_test1=np.empty(shape=(1531,64,64,3))

for i in range(1532):

    im=io.imread(test_path+str(i+1)+'.jpg')

    X_test1[i]=transform.resize(im,output_shape=(64,64,3))
y_pred1=cnn.predict(X_test1)