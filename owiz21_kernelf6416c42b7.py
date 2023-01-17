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
import matplotlib.pyplot as plt

import PIL

from PIL import Image



df=pd.read_csv('../input/train.csv')

x_train=df.iloc[:,0].values

y_train=df.iloc[:,1].values

print("done importing ")

x_t=[]

for i in range(0,17500):

    x_t.append(np.asarray(Image.open("../input/train/train/"+x_train[i])))

print("creating train dataset")
x_t=np.array(x_t)
df=pd.read_csv("../input/sample_submission.csv")

print("getting test ids")
x_test=df.iloc[:,0].values

x_tes=[]

for i in range(0,4000):

    x_tes.append(np.asarray(Image.open("../input/test/test/"+x_test[i])))

x_test=np.array(x_tes)



x_t = x_t.astype('float32')

x_test = x_test.astype('float32')



x_t /= 255

x_test /= 255

print("standardizing and normalizing the test and train datasets")
from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

print("importing dependencies for the CNN")
classifier = Sequential()



# Step 1 - Convolution

classifier.add(Convolution2D(32, 3, 3, input_shape = (32, 32, 3), activation = 'relu'))



# Step 2 - Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Adding a second convolutional layer

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Step 3 - Flattening

classifier.add(Flatten())



# Step 4 - Full connection

classifier.add(Dense(output_dim = 128, activation = 'relu'))

classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))



# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

print("CNN compiled ")
classifier.fit(x=x_t,y=y_train,epochs=25)

pred=classifier.predict(x_test)

ids=df.iloc[:,0].values

pred=pred>=.5

pred=np.array(pred)

pred=pred.astype('int16')

print("predicting the results")
out_dict={'id':ids,'has_cactus':pred[:,0]}

out=pd.DataFrame(out_dict)

out.to_csv("output2.csv",index=False)

print("writing output to file")