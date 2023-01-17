# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras



from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import BatchNormalization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv');

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv');
rows = 28

cols = 28

tot_rows = train.shape[0]

X_train = train.values[:,1:]

y_train = keras.utils.to_categorical(train.label, 10)

X_train = X_train.reshape(tot_rows, rows, cols, 1)/255.0



X_test = test.values[:]

test_num_img = test.shape[0]

X_test = X_test.reshape(test_num_img, rows, cols, 1)/255.0
classifier = Sequential()



#1st layer

classifier.add(Conv2D(32,(5,5),input_shape=(28,28,1),activation = 'relu',padding='same'))

classifier.add(BatchNormalization())

classifier.add(Conv2D(32,(3,3),activation = 'relu',padding='same'))

classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(pool_size=(2,2), strides=None))

classifier.add(BatchNormalization())

classifier.add(Dropout(0.25))



#2st layer

classifier.add(Conv2D(64,(5,5),activation = 'relu',padding='same'))

classifier.add(BatchNormalization())

classifier.add(Conv2D(64,(3,3),activation = 'relu',padding='same'))

classifier.add(BatchNormalization())

classifier.add(Conv2D(64,(3,3),strides=(2,2),activation = 'relu',padding='same'))

classifier.add(BatchNormalization())

classifier.add(Dropout(0.25))



classifier.add(Flatten())



#connection

classifier.add(Dense(units=128,activation='relu'))

classifier.add(Dropout(0.4))

classifier.add(Dense(units=10,activation='softmax'))
classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])



classifier.fit(X_train,y_train,epochs=100,batch_size=64,validation_split=0.1,shuffle=True)
result = classifier.predict_classes(X_test)
out = pd.DataFrame({"ImageId": i+1 , "Label": result[i]} for i in range(0, test_num_img))

out.to_csv('submission.csv', index=False)