#Importing Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
#Loading Train_Set

df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_train.head()
#Split pixel data & label

X_train = df_train.iloc[:,1:].values

y_train = df_train.iloc[:,0].values
#Visual of the 1st image

first_image = X_train[0].reshape(28,28)

plt.imshow(first_image,cmap='gist_gray')
print(X_train.shape)

print(X_train.min())

print(X_train.max())
#Change dimension of all the images

#For a single image, instead of the pixels being in an array(784,1) we prefer it in the format (28,28)

#For 42000 images ~ (42000,28,28) and For gray scale ~ (42000,28,28,1)

#Had it been a coloured image, the dimesion would have been ~ (42000,28,28,3)[3 for RGB]

#To normalize the magnitudes of the pixels we diivide all the pixels by 255

X_train = X_train.reshape(-1,28,28,1)/255

print(X_train.shape)

print(X_train.min())

print(X_train.max())
#One_Hot_Encode (using Keras API instead of sklearn)

y_train = tf.keras.utils.to_categorical(y_train)

print(y_train)
#Initialise CNN

cnn = tf.keras.models.Sequential()
#Convolution Layer

cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=5,activation='relu',input_shape=[28,28,1]))



#Max-Pooling

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))



#Second Conolution Layer

cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=5,activation='relu'))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
#Flattening

cnn.add(tf.keras.layers.Flatten())



#Fully Connected Layer

cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))



#Hidden Layer

cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))



#Output Layer

cnn.add(tf.keras.layers.Dense(units=10,activation='softmax'))



#Compilation

cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#Training the Model

cnn.fit(X_train,y_train,batch_size=32,epochs=5)
#Loading the Test Set

X_test = pd.read_csv('../input/digit-recognizer/test.csv')
#Prediction

y_prob = cnn.predict(X_test.iloc[:,:].values.reshape(-1,28,28,1)/255)

pred_digit = np.argmax(y_prob, axis=1)

pred_digit = pd.Series(pred_digit,name='Label')

print(pred_digit)
#Submission for Kaggle Competition

pred = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pred_digit],axis = 1)

pred.to_csv("submission.csv",index=False)