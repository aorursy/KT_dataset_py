import pandas as pd

from tensorflow import keras

from keras import  Sequential

from keras.layers import Flatten,Dense

import os
train = pd.read_csv("../input/Kannada-MNIST/train.csv") 

test = pd.read_csv("../input/Kannada-MNIST/test.csv")
for dirname,_,filesNames in os.walk("../input/Kannada-MNIST/"):

    print(filesNames)
%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from matplotlib.pyplot import imread, imshow, subplots, show

TRAINING_DIR , TESTING_DIR  =  "../input/Kannada-MNIST/","../input/Kannada-MNIST/"



# Parameters for our graph; we'll output images in a 4x4 configuration

nrows = 4

ncols = 4



# Index for iterating over images

pic_index = 0



try:

    # Set up matplotlib fig, and size it to fit 4x4 pics

    fig = plt.gcf()

    fig.set_size_inches(ncols * 4, nrows * 4)

    pic_index += 8



    next_train_pix = [os.path.join(TRAINING_DIR, fname) for fname in os.listdir('../input/Kannada-MNIST/')[pic_index - 8:pic_index]]

    next_test_pix = [os.path.join(TESTING_DIR, fname) for fname in os.listdir('../input/Kannada-MNIST/')[pic_index - 8:pic_index]]



    for i, img_path in enumerate(next_train_pix + next_test_pix):

        # Set up subplot; subplot indices start at 1

        sp = plt.subplot(nrows, ncols, i + 1)

        sp.axis('On')  # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)

        plt.imshow(img)



    plt.show()

    

except:

    pass
X = train.drop(columns=['label'])

Y =  train['label']
from sklearn.model_selection import train_test_split
X_train,X_test , Y_train , Y_test = train_test_split(X,Y ,test_size=0.2,random_state=42)
print("X_train:: ",X_train.shape)

print("X_test:: ",X_test.shape)

print("Y_train:: ",Y_train.shape)

print("Y_test:: ",Y_test.shape)
#normalize the data

train = train/255

test = test/255
model = Sequential()

model.add(Dense(50, input_dim=784, activation='relu'))

model.add(Dense(40, activation='relu'))

model.add(Dense(30, activation='relu'))

model.add(Dense(20, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss = "sparse_categorical_crossentropy",metrics=['accuracy'])
model.fit(X_train,Y_train,validation_data=(X_test,Y_test),verbose=1,epochs=100,batch_size=200)
import numpy as np

test = test.drop(columns='id')

lables = model.predict_classes(test)
data = pd.DataFrame()

for i in range(len(lables)):

    index = pd.DataFrame([i])

    value =  pd.DataFrame([i])

    row = pd.concat([index,value],axis=1)

    data = pd.concat([data,row],axis=0)
data.columns=['id','label']
data.head(600)
submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

submission['label'] = lables

submission.to_csv("submission.csv",index=False)
