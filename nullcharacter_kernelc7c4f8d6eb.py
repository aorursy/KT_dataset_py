

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(2)



dataset = pd.read_csv("../input/digit-recognizer/train.csv")

y = dataset.iloc[:,0].values

x = dataset.iloc[:,1:].values.reshape(-1,28,28,1)



x = x.astype('float32')

y = y.astype('float32')



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0,test_size = 0.2)

x_train = x_train/255.0

x_test = x_test/255.0
x_train.shape
plt.imshow(x_train[111][:,:,0])
import tensorflow as tf

from tensorflow import keras

from keras.models import  Sequential

from keras.layers.core import  Lambda , Dense, Flatten, Dropout

from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D



def CNN_model():

    model = Sequential([

        Convolution2D(32,(3,3), activation='relu', input_shape=(28,28,1)),

        Convolution2D(32,(3,3), activation='relu'),

        MaxPooling2D(),

        Dropout(0.25),

        Convolution2D(64,(3,3), activation='relu'),

        Convolution2D(64,(3,3), activation='relu'),

        MaxPooling2D(),

        Flatten(),

        Dense(256, activation='relu'),

        Dense(10, activation='softmax')

        ])

    model.compile(optimizer = 'Adam',

             loss = 'sparse_categorical_crossentropy',

             metrics = ['accuracy'])

    return model
model = CNN_model()
model.fit(x_train,y_train,epochs = 30)
y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis = 1)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test,y_pred)

print(acc)
test = pd.read_csv("../input/digit-recognizer/test.csv")

test = test.astype('float32')

test = test/255

test = test.iloc[:,:].values.reshape(-1,28,28,1)

y_d = model.predict(test)

y_d = np.argmax(y_d,axis = 1)

print(y_d)

sub = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

my_submission = pd.DataFrame({'ImageId':sub.ImageId,'Label': y_d})

# you could use any filename. We choose submission here

print(my_submission.to_csv('submission.csv', index=False))


