# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow import keras 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
img_rows, img_cols = 28,28

num_classes = 10

def clean_and_prep(raw,test=False):

    if test==False:

        y=raw[:,0];

        out_y = keras.utils.to_categorical(y,num_classes)



        x = raw[:,1:]

        num_images = raw.shape[0]

        out_x = x.reshape(num_images,img_rows,img_cols,1)

        out_x = out_x/255;

        return out_x,out_y

    else :

        x = raw

        num_images = raw.shape[0]

        out_x = x.reshape(num_images,img_rows,img_cols,1)

        out_x = out_x/255;

        return out_x;
digit_file = "../input/train.csv";

digit_data = np.loadtxt(digit_file,skiprows=1,delimiter=',');

x,y = clean_and_prep(digit_data)
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Flatten,Conv2D,Dropout

model = Sequential()

model.add(Conv2D(16,kernel_size=(3,3),activation='relu',input_shape=(img_rows,img_cols,1)))

model.add(Dropout(rate=0.3))

model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))

model.add(Dropout(rate=0.3))

model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))

model.add(Dropout(rate=0.3))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dense(num_classes,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.fit(x,y,epochs=7,batch_size=10)
test_file = "../input/test.csv";

test_data = np.loadtxt(test_file,skiprows=1,delimiter=',')

test_x = clean_and_prep(test_data,True)
preds = model.predict(test_x)
predicted_values = preds.argmax(axis= -1)

index = np.empty((0),dtype='int64')

for i in range(len(predicted_values)):

    index= np.append(index,i+1);
output = pd.DataFrame({'ImageId': index[:],

                     'Label': predicted_values[:]})
output.to_csv('submission.csv',index=False)
pd.read_csv('./submission.csv')