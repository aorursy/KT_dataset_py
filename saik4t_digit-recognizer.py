import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K
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
#Reading CSV train data

data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

#first peak in the dataset

data.head(10)



#extract train data

datasetTrain = data.drop(['label'], axis=1)

#peak at the data

datasetTrain 
#converting to a list

x_data = datasetTrain.values.tolist()

# converting to a list

y_data = data.label.tolist()
#Splitting test and train data #70% as train data and 30% as test data # len(x_data) = len(y_data)

x_train = x_data[:(int)(len(x_data)*.7)] 

y_train =  y_data[:(int)(len(x_data)*.7)] 

x_test = x_data[(int)(len(x_data)*.7):] 

y_test = y_data[(int)(len(x_data)*.7):] 



# reshaping train and test data in format (28,28,1) which is (width,height,channel)

x_train = [np.array(x).reshape(-1,28,1) for x in x_train]

x_test = [np.array(x).reshape(-1,28,1) for x in x_test]
#converting data tofloat32 and scaling the values between [0-1] by dividing by 255

x_train = np.array( x_train, dtype=np.float32)

x_test = np.array(x_test, dtype=np.float32)

x_train /= 255

x_test /= 255

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, 10)

y_test = keras.utils.to_categorical(y_test, 10)
#sequential model creation in KERAS

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28,28,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
#Compile Parameters

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])
#Train th enetwork

model.fit(x_train, y_train,

          batch_size=32,

          epochs=100,

          verbose=1,

          validation_data=(x_test, y_test))
#load Test Data

testData = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

#convert test data

x_validation = testData.values.tolist()

x_validation = [np.array(x).reshape(-1,28,1) for x in x_validation]
x_validation = np.array( x_validation, dtype=np.float32)

x_validation /=255

print('x_train shape:', x_validation.shape)

#List of prediction

result_list = model.predict(x_validation)
result_list =  np.array(result_list, dtype = np.float16)
#check some of the test Images if they are working visually

image_num = 4

prediction = result_list[image_num]

label = prediction.tolist().index(max(prediction))

print(label)



e = x_validation[image_num].reshape(-1,28)

import matplotlib.pyplot as plt

import numpy as np

 

#X = np.random.random((100, 100)) # sample 2D array

plt.imshow(e, cmap="gray")

plt.show()
#Create output file according the requirment 

output = []

for x in range(len(result_list)):

    prediction = result_list[x]

    label = prediction.tolist().index(max(prediction))

    output.append([x+1,label])
#convert it to pandas dataframe

df_output = pd.DataFrame(output, columns = ['ImageId', 'Label'])

#sneak peak

df_output
#Save file

df_output.to_csv('output2.csv',index = False)