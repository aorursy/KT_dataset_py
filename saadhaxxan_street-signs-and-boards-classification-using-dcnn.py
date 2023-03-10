import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#importing pickle module to unpickle the files in the dataset



import pickle



with open('/kaggle/input/germantrafficsigns/train.p','rb') as f:

  train_data=pickle.load(f)

with open('/kaggle/input/germantrafficsigns/test.p','rb') as f:

  test_data=pickle.load(f)

with open('/kaggle/input/germantrafficsigns/valid.p','rb') as f:

  val_data=pickle.load(f)

#splitting the data into our variables



x_train, y_train = train_data['features'],train_data['labels']

x_test, y_test = test_data['features'],test_data['labels']

x_val, y_val = val_data['features'],val_data['labels']
#showing the size of each variable we have



print(x_train.shape)

print(x_test.shape)

print(x_val.shape)
import pandas as pd



data = pd.read_csv('/kaggle/input/germantrafficsigns/signnames.csv')



print(data)



#from the dataframe printed below we come to know that the dataset has 43 classes.
#import matplotlib for visualizing images



import matplotlib.pyplot as plt



#the block of code below just display an image from our data



plt.imshow(x_train[0])

print(x_train[0].shape)
#converting images into gray scale so that the neural network can learn the pattern easily



import cv2



def gray(img):

  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  return img



#equalizing images to make the features in the images more porminent for the model to understand



def equalize(img):

  img = cv2.equalizeHist(img)

  return img



def preprocessing(img):

  img = gray(img)

  img = equalize(img)

  #now normalizing the images

  img = img/255

  return img

  
#using map fucntion to iterate through the whole dataset and apply our preprocessing fucntion to every image

import numpy as np



x_train= np.array(list(map(preprocessing,x_train)))

x_val= np.array(list(map(preprocessing,x_val)))

x_test= np.array(list(map(preprocessing,x_test)))
#showing the new preprocessed images



plt.imshow(x_train[0])

print(x_train[0].shape)
#converting the labels into categorical variables

from keras.utils.np_utils import to_categorical



y_cat_train = to_categorical(y_train, 43)

y_cat_test = to_categorical(y_test, 43)

y_cat_val = to_categorical(y_val, 43)





#reshaping the images



x_train = x_train.reshape(34799, 32, 32, 1)

x_test = x_test.reshape(12630, 32, 32, 1)

x_val = x_val.reshape(4410, 32, 32, 1)



print(x_train.shape)



#importing keras and required layers to create the model

import keras

from keras.models import Sequential

from keras.optimizers import Adam

from keras.layers import Dense

from keras.layers import Flatten, Dropout

from keras.layers.convolutional import Conv2D, MaxPooling2D



# create model

model = Sequential()

model.add(Conv2D(30, (5, 5), input_shape=(32, 32, 1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(15, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(500, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(43, activation='softmax'))

  



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



print(model.summary())
model.fit(x_train,y_cat_train,epochs=20,batch_size=400,verbose=1,shuffle=1)
from sklearn.metrics import classification_report  
prediction = model.predict_classes(x_test)
print(classification_report(y_test,prediction))
model.save('street_signs.h5')