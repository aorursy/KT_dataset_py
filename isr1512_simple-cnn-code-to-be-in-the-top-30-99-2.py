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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print('Train contains %d rows and %d columns'%(train.shape[0],train.shape[1]))

print('Test contains %d rows and %d columns'%(test.shape[0],test.shape[1]))
train.head()
test.head()
train_data = train.drop(['label'],axis=1)

train_label = train[['label']]
train_img = np.array(train_data).reshape(-1,28,28,1)

test_img = np.array(test).reshape(-1,28,28,1)
print(train_img.shape,test_img.shape)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
fig, ax = plt.subplots(3,3,figsize = (8,8))

for i, j in enumerate(ax.flat):

    j.imshow(train_img[i][:,:,0], cmap ='binary')

    j.text(2,2,str(train_label.label[i]),color = 'red')
from keras.preprocessing import image

from keras.layers.normalization import BatchNormalization
batch_size = 32

np_classes = 10

nb_epoch = 10

#Normalize the data

x_train = train_img.astype('float')

x_test = test_img.astype('float')

x_train /= 255

x_test /= 255
from keras.utils import to_categorical

y_train = to_categorical(train_label, np_classes)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Convolution2D, MaxPooling2D,Conv2D, MaxPool2D
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size=(3,3), padding='same', input_shape = (28,28,1),activation='relu'))

model.add(Conv2D(filters = 32, kernel_size=(3,3), activation='relu'))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.4))



model.add(BatchNormalization())

model.add(Conv2D(filters=64,kernel_size=(2,2), padding = 'same', activation = 'relu'))

model.add(Conv2D(filters=64, kernel_size=(2,2), padding = 'same', activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.4))



model.add(Flatten())

model.add(BatchNormalization())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(np_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])
model.summary()
model.fit(x_train, y_train,batch_size = batch_size, epochs = nb_epoch, shuffle = True)
pred = model.predict_classes(x_test).reshape(-1).astype(np.int8)
submis = pd.read_csv('../input/sample_submission.csv')
submis['Label'] = pred
submis.to_csv('digit_recog_v1.csv', index=False)