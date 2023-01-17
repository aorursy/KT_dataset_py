# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# these are usuful libs for modeling

from keras import models, layers

from keras.utils import to_categorical

from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Dense, Lambda

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore") # suppress warnings

import tensorflow as tf

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# reading csv files

train_data = pd.read_csv(r'/kaggle/input/digit-recognizer/train.csv')

test_data = pd.read_csv(r'/kaggle/input/digit-recognizer/test.csv')
print(train_data.shape)

print(test_data.shape)
# scaling the value in 0-1 , so /255. 

X = train_data.iloc[:,1:]

X /= 255

#X = np.array(X, dtype='float32')

print("X matrix shape:", X.shape)



y = train_data.iloc[:,0]

#y = np.array(y, dtype='float32')

print("y matrix shape:", y.shape)



test = test_data/255

#test = np.array(test, dtype='float32')
X = X.values.reshape(train_data.shape[0],28,28,1)

test = test.values.reshape(test_data.shape[0],28,28,1)
# Plot examples of the data.

plt.figure(1, figsize=(14,3))

for i in range(10):

    plt.subplot(1,10,i+1)

    plt.imshow(X[i].reshape(28,28), cmap='gray', interpolation='nearest')

    plt.xticks([])

    plt.yticks([])
# one hot of target values using keras's to_categorical class

y = to_categorical(y)



# splits train/test set 

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size =0.3,random_state=29)
Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape
# need BatchNormalization, normalize the batch

mean_px = Xtrain.mean().astype(np.float32)

std_px = Xtrain.std().astype(np.float32)



def standardize(x): 

    return (x-mean_px)/std_px
# this is a sequential model with the classic keras CNN architecture with batch added layers of: 

#normalization, Conv2D, MaxPooling, Flatten, Dense, Dropout

#rectified linear unit 'relu' and 'softmax' for activation

#adaptive moment estimation 'adam' for optimizer (rmsprop not the best to use for this scenario)

#categorical cross entropy performs better than binary loss function

#

def digit_model():

    model = models.Sequential()

    model.add(Lambda(standardize,input_shape=(28,28,1)))

    model.add(Convolution2D(32,(3,3), activation = 'relu'))

    model.add(BatchNormalization(axis=1))   

    model.add(Convolution2D(64,(3,3), activation = 'relu'))

    model.add(MaxPooling2D())

    model.add(Convolution2D(128,(3,3), activation = 'relu'))

    model.add(BatchNormalization(axis=1))

    model.add(Convolution2D(128,(2,2), activation = 'relu'))

    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.4))

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.3))

    model.add(Dense(10, activation='softmax'))          

    model.compile(optimizer='adam', loss='categorical_crossentropy',

                  metrics=['accuracy'])

    return model   
classifier = digit_model()

# traing the model with 10 epochs and 1000 batch size

classifier.fit(Xtrain, ytrain, epochs=40,batch_size=1000,validation_data=(Xtest,ytest))
#prediction of submission_test set

prediction = classifier.predict(test)

predictions = np.argmax(prediction, axis=1)
# submission

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})



submissions.shape
import os

os.chdir(r'/kaggle/working')



submissions.to_csv('sub20.csv', index=False)

#print("save success")



from IPython.display import FileLink

FileLink(r'sub20.csv')