# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

# #         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
imagepaths = []

for root,dirs,files in os.walk("/kaggle/input/soli-data/dsp/",topdown=False):

    for name in files:

        path = os.path.join(root,name)

        if path.endswith("h5"):

            imagepaths.append(path)
print(len(imagepaths))
import h5py

import numpy as np
raw_data=np.empty((1,1024))#creating enpty np

labels=np.empty(1)

for hfile in imagepaths[:100]: # loading only 10 data

    with h5py.File(hfile,'r') as f:

        for channel in range(4):

            x=f['ch{}'.format(channel)][()]

            y=f['label'][()]

           

            raw_data=np.vstack((raw_data,x)) # stacking row wise

            labels=np.vstack((labels,y))



raw_data=np.delete(raw_data,0,0)# delete first element

labels=np.delete(labels,0,0)
print(raw_data.shape, labels.shape)

data=raw_data.reshape(raw_data.shape[0],32,32,1)
from sklearn.model_selection import train_test_split
X_train,X_val_test,y_train, y_val_test=train_test_split(data,labels,test_size=.5)
X_test,X_val,y_test,y_val=train_test_split(X_val_test,y_val_test,test_size=0.5)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense,Flatten
model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,1)))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3),activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dense(512,activation='relu'))

model.add(Dense(12,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size= 32, verbose = 2, validation_data = (X_val, y_val))

model.evaluate(X_test,y_test)