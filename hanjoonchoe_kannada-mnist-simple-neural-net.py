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



import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from sklearn.utils import shuffle

from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
train_data = train.loc[:,'pixel0':].values

train_data = train_data/255.0

train_data=train_data.reshape(-1,28,28,1)

test_data = test.loc[:,'pixel0':].values

test_data = test_data/255.0

test_data=test_data.reshape(-1,28,28,1)

train_label=train['label']

train_label=train_label.values
train_data, train_label = shuffle(train_data, train_label, random_state = 5)
input_shape = (28, 28, 1)

# Creating a Sequential Model and adding the layers

model = Sequential()

model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # Flattening the 2D arrays for fully connected layers

model.add(Dense(128, activation=tf.nn.relu))

model.add(Dropout(0.2))

model.add(Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy', 

              metrics=['accuracy'])

model.fit(x=train_data,y=train_label, epochs=10)
model.evaluate(train_data, train_label)
predictions = model.predict(test_data)

prediction1 = np.argmax(predictions,axis=1)
#Create a  DataFrame

submission = pd.DataFrame({'id':test['id'],'label':prediction1.astype(int)})



#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'mnist2.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)