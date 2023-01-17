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
train_data=pd.read_csv('..//input//train.csv')

test_data=pd.read_csv('..//input//test.csv')
x_train=train_data.drop('label',axis=1).values

y_train=train_data['label'].values
x_train=x_train.reshape(42000,28,28)
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(x_train[i],cmap=plt.cm.gray)

    plt.xlabel(y_train[i])
#Normalization

x_train=x_train/255

x_test=test_data/255
import keras

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten
#Converting labels in to categorical variables

y_train=to_categorical(y_train)
#Adding Learning Rate Decay Regularization 

from keras.optimizers import SGD

lr=0.01

epochs=100

decay_rate=lr/epochs

sgd=SGD(lr=lr,momentum=0.8,decay=decay_rate)



model3=Sequential()

model3.add(Flatten(input_shape=(28,28)))

model3.add(Dense(256,activation='relu'))

model3.add(Dense(128,activation='relu'))

model3.add(Dense(32,activation='relu'))

model3.add(Dense(10,activation='softmax'))



model3.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

history_3=model3.fit(x_train,y_train,batch_size=128,epochs=10,verbose=0,validation_split=0.33,shuffle=True)
model3.evaluate(x_train,y_train)
x_test=test_data.values.reshape(28000,28,28)
results = model3.predict(x_test)

results = np.argmax(results,axis = 1)

data_out = pd.DataFrame({'ImageId': range(1, len(x_test) + 1), 'Label': results})
data_out.to_csv('MNIST-Prediction.csv', index = None)
plt.figure(figsize=(10,10))

for i in range (25):

    plt.subplot(5,5,1+i)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(x_test[i],cmap=plt.cm.gray)

    plt.xlabel(np.argmax(y_predict[i]))