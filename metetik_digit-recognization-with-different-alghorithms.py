import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')



#for logistic regression

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



#for Artificial Neural Networks

import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout ,Flatten, Conv2D, MaxPool2D

from sklearn.metrics import confusion_matrix

import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
train = pd.read_csv('../input/train.csv')



test = pd.read_csv('../input/test.csv')
train.head()
X = train.iloc[:,1:]



Y = train['label']
score = [] # we will put scores into this serie



#we will split data to evaluate model

x_train , x_val , y_train , y_val = train_test_split(X, Y, test_size = 0.15, random_state = 41)



#we will try model for each number of iteration

for i in range(1,15): 

    lr = linear_model.LogisticRegression(max_iter = i,random_state = 41)

    lr.fit(x_train,y_train) 

    score.append(lr.score(x_val,y_val))



#plot of scores and parameters

plt.figure(figsize = (15,9)) 

plt.ylabel('Prediction Score') 

plt.xticks(np.arange(0, 15, 1)) 

plt.xlabel('Number of Parameters') 

plt.plot(range(1,15),score) 

plt.show()
lr = linear_model.LogisticRegression(max_iter = 9,random_state = 41)



lr.fit(x_train,y_train)



lr.score(x_val,y_val)
test.head()
xtrain = X.iloc[:33600].values/255



xtest = X.iloc[33600:41999].values/255



ytrain = Y.iloc[:33600].values



ytest = Y.iloc[33600:41999].values



ytrain = keras.utils.to_categorical(ytrain, 10)



ytest = keras.utils.to_categorical(ytest, 10)
model = Sequential()

model.add(Dense(700,activation = 'tanh',input_shape=(784,)))

model.add(Dropout(0.3))

model.add(Dense(100,activation = 'relu'))

model.add(Dropout(0.3))

model.add(Dense(10,activation = 'relu'))

model.add(Dense(10, activation='softmax'))



model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.fit(xtrain, ytrain,

          epochs=25,

          batch_size=500)

score = model.evaluate(xtest, ytest, batch_size=500)
model = Sequential()



model.add(Conv2D(filters = 64,kernel_size=(4,4),padding='same',activation='relu',input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.2))

model.add(Dense(128, activation = "relu"))

model.add(Dropout(0.2))

model.add(Dense(128, activation = "relu"))

model.add(Dense(10, activation = "softmax"))
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])



model.fit(xtrain.reshape(-1,28,28,1), ytrain,

          epochs=64,

          batch_size=512)

score = model.evaluate(xtest.reshape(-1,28,28,1), ytest, batch_size=500)
p = model.predict(test.values.reshape(-1,28,28,1)/255)



p = np.argmax(p,axis = 1)
plt.figure(figsize = (15,15))

for i in range (1,101):

    plt.subplot(10,10,i)

    plt.imshow(test.values[i-1].reshape(28,28),cmap = 'Greys')

    plt.axis('off')

    plt.title(('Label : ',p[i-1]))

df = {'ImageId' : range(1,28001),'Label' : p}



submission = pd.DataFrame(df)



submission.to_csv('submission.csv', index = False)