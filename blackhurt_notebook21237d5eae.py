# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_dir=pd.read_csv('/kaggle/input/gtsrb-german-traffic-sign/Train.csv')

test_dir=pd.read_csv('/kaggle/input/gtsrb-german-traffic-sign/Test.csv')

train=train_dir['Path'].tolist()
y_train=train_dir['ClassId']

test=test_dir['Path'].tolist()
y_train=y_train.values.reshape(39209,1)
import cv2

training=[]

for i in range(0,39209):

    source='/kaggle/input/gtsrb-german-traffic-sign/'+train[i]

    img=cv2.imread(source,1)

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img=cv2.resize(img,(30,30),interpolation=cv2.INTER_NEAREST)

    training.append(img)

    
len(test_dir)
import cv2

testin=[]

for i in range(0,12630):

    source='/kaggle/input/gtsrb-german-traffic-sign/'+test[i]

    img=cv2.imread(source,1)

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img=cv2.resize(img,(30,30),interpolation=cv2.INTER_NEAREST)

    testin.append(img)

    
testin=np.array(testin)
training.shape
y_label=train_dir['ClassId'].tolist()

y_label
model=keras.layers.Sequential([kera])
from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout



model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(30,30,3)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(43, activation='softmax'))



#Compilation of the model

model.compile(

    loss='sparse_categorical_crossentropy', 

    optimizer='adam', 

    metrics=['accuracy']

)

history = model.fit(training,y_label, batch_size=32, epochs=30, verbose=2)
y_label=np.array(y_label)
pred=model.predict(testin)
from sklearn.metrics import accuracy_score

accuracy_score(pred,testin)