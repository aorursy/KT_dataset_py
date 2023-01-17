# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')

train_im=np.load('../input/trainbeg.npy')/255
train_imlabel=np.load('../input/trainLabels.npy')
# train_im=train_im[:9000]
# train_imlabel=train_imlabel[:9000]
print(train_im.shape)
test_im=np.load('../input/testbeg.npy')/255
plt.imshow(train_im[0])
lab=[y[0] for y in train_imlabel]
labels=[]
for i in lab:
    f=np.zeros(30)
    f[i]=1
    labels.append(f)
print(labels[0])
labels=np.array(labels)
labels.shape[1]
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
#We create a Sequential model using 'categorical cross-entropy' as our loss function and 'adam' as the optimizer.

model = Sequential()
model.add(Convolution2D(32, (3,3), activation='relu', padding='same',input_shape = (64,64,3)))
#if you resize the image above, change the input shape
model.add(Convolution2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(30, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()
early_stops = EarlyStopping(patience=3, monitor='val_acc')
model.fit(train_im, labels, batch_size=100, epochs=8, validation_split=0.3, callbacks=[early_stops])
predictions = model.predict(test_im)
predictions[0]
trainmain=pd.read_csv('../input/sample_submission.csv')
trainmain.drop('image_id',1).columns
# test['Image_id']
pr2=pd.DataFrame(data=test['Image_id'],columns=['Image_id'])
pr2['image_id']=test['Image_id']
pr2.drop('Image_id',axis=1,inplace=True)


pr=pd.DataFrame(data=predictions,columns=trainmain.drop('image_id',1).columns)
# pr2=pd.DataFrame(data=test['Image_id'],columns=['image_id'])
mmm=pd.concat([pr2,pr],axis=1)
mmm.head()
mmm.to_csv('p3.csv',index=False)
