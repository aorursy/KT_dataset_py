# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dftrain=pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

dftest=pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

dfdig=pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')



X_train=dftrain.values

X_test=dftest.values

y_train=X_train[:,0]

X_train=X_train[:,1:]

label_test=X_test[:,0]

X_test=X_test[:,1:]
X_dig=dfdig.values

y_dig=X_dig[:,0]

X_dig=X_dig[:,1:]
import matplotlib.pyplot as plt

i=25

print(y_train[i])

plt.imshow(X_train[i,:].reshape(28,28),cmap='gray')
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization,GaussianNoise,Activation

from keras.layers.convolutional import MaxPooling2D

from keras.models import Sequential

from keras.utils import to_categorical

from keras.optimizers import Adam
#Dig for Validation



X_train = X_train.reshape((-1, 28, 28, 1))

X_test = X_test.reshape((-1, 28, 28, 1))

X_dig = X_dig.reshape((-1, 28, 28, 1))



y_train = to_categorical(y_train,10)

#y_test = to_categorical(y_test, 10)

y_dig = to_categorical(y_dig, 10)

y_dig.shape
model = Sequential()

#model.add(GaussianNoise(0.1, input_shape=(28,28,1)))

#model.add(Conv2D(32, kernel_size=(7,7), padding='same', activation='relu'))

#model.add(MaxPooling2D((2,2), padding='same'))

#model.add(Conv2D(32, kernel_size=(7,7), padding='same', activation='relu'))

#model.add(MaxPooling2D((2,2), padding='same'))

#model.add(Conv2D(64, kernel_size=(5,5), padding='same', activation='relu'))

#model.add(MaxPooling2D((2,2), padding='same'))

#model.add(Conv2D(32, kernel_size=(5,5), padding='same', activation='relu'))



#model.add(MaxPooling2D((2,2), padding='same'))



model.add(Conv2D(16, kernel_size=(7,7), padding='same',  input_shape=(28,28,1),activation='relu'))

model.add(MaxPooling2D((2,2), padding='same'))

model.add(Conv2D(16, kernel_size=(7,7), padding='same', activation='relu'))

model.add(MaxPooling2D((2,2), padding='same'))



#model.add(BatchNormalization(momentum=0.15))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(32, activation='relu'))

#model.add(GaussianNoise(0.1))

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])#Adam(lr=0.001,beta_1=0.9,beta_2=0.999)

model.summary()
import keras

es=keras.callbacks.EarlyStopping(monitor='val_loss', verbose=0, mode='auto', restore_best_weights=True)
model.fit(x=X_train,y=y_train,batch_size=512,epochs=20,verbose=1,validation_data=(X_dig, y_dig), callbacks=[es])
'''i=20

hist=[]

for _ in range(i):

    model.fit(x=X_train,y=y_train,batch_size=512,epochs=20,verbose=1,validation_data=(X_dig, y_dig), callbacks=[es])

    loss,acc=model.evaluate(X_dig, y_dig)

    

    hist.append(acc)'''
y_pred=model.predict(X_test)
label_pred = np.argmax(y_pred, axis = 1)

sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
ids = sample_sub['id'].to_list()

file = open("submission.csv", "w")

file.write("id,label\n")

for id_, pred in zip(ids, label_pred):

    file.write("{},{}\n".format(id_, pred))

file.close()