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
import os

os.mkdir('/kaggle/temp/')

os.mkdir('/kaggle/temp/train/')

os.mkdir('/kaggle/temp/validation/')

os.mkdir('/kaggle/temp/train/0/')

os.mkdir('/kaggle/temp/train/1/')

os.mkdir('/kaggle/temp/validation/0/')

os.mkdir('/kaggle/temp/validation/1/')
import os

print(os.listdir('/kaggle/temp/'))
import os

import numpy as np

import imageio

import shutil

path=os.path.join('/kaggle/input/breast-histopathology-images/')





dirs=os.listdir(path)



for dir in dirs:

    src=path+dir+'/0/'

    #print(src)

    if not os.path.exists(src):

        continue

    i=0

    flag=0

    dstn='/kaggle/temp/train/0/'

    for file in os.listdir(src):

        valid_size=int(0.9*len(os.listdir(src)))

        if(i==valid_size):

            dstn='/kaggle/temp/validation/0/'

        image=imageio.imread(src+file+'/')

        if(image.shape!=(50,50,3)):

            continue

        if flag==0:

            shutil.copy2(src+file,dstn)

            flag=1

        else:

            flag=0

        i+=1

    

    src=path+dir+'/1/'

    i=0

    dstn='/kaggle/temp/train/1/'

    for file in os.listdir(src):

        valid_size=int(0.9*len(os.listdir(src)))

        if(i==valid_size):

            dstn='/kaggle/temp/validation/1/'

        image=imageio.imread(src+'/'+file+'/')

        if(image.shape!=(50,50,3)):

            continue

        shutil.copy2(src+file,dstn)

        i+=1
import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,BatchNormalization,Dropout



model=Sequential([

    Conv2D(32,kernel_size=3,input_shape=(50,50,3),activation='relu'),

    MaxPool2D((2,2)),

    Dropout(0.2),

    BatchNormalization(),

    Conv2D(32,kernel_size=3,activation='relu'),

    MaxPool2D((2,2)),

    Dropout(0.2),

    BatchNormalization(),

    Conv2D(64,kernel_size=3,activation='relu'),

    MaxPool2D((2,2)),

    Dropout(0.2),

    BatchNormalization(),

    Conv2D(32,kernel_size=3,activation='relu'),

    MaxPool2D((2,2)),

    Dropout(0.2),

    BatchNormalization(),

    Flatten(),

    Dense(128,activation='relu'),

    Dense(2,activation='sigmoid')

])
model.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dirc='/kaggle/temp/train/'

train_gen=ImageDataGenerator(rescale=1/255.0)

train_data_gen=train_gen.flow_from_directory(dirc,target_size=(50,50),batch_size=32)
dirc='/kaggle/temp/validation/'

valid_gen=ImageDataGenerator(rescale=1/255.0)

valid_data_gen=valid_gen.flow_from_directory(dirc,target_size=(50,50),shuffle=False,batch_size=32)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history=model.fit_generator(train_data_gen,epochs=10,validation_data=valid_data_gen)
from sklearn.metrics import confusion_matrix



valid_data_gen.reset



y_pred=model.predict_generator(valid_data_gen,steps=528)

y_pred=np.argmax(y_pred,axis=1)

labels=valid_data_gen.classes

cm=confusion_matrix(labels,y_pred)
print(cm)
precision=cm[0,0]/(cm[0,0]+cm[0,1])

recall=cm[0,0]/(cm[0,0]+cm[1,0])

fscore=2*(precision*recall)/(precision+recall)

print(precision)

print(recall)

print(fscore)