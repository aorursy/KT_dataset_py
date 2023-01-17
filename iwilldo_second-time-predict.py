# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print("the  train_image has picture： %d \nthe test_image has picture:%d" % (len(os.listdir('../input/train_images')),

                                                                           len(os.listdir('../input/test_images'))))



# Any results you write to the current directory are saved as output.
#import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical

import cv2



train_data=pd.read_csv("../input/train.csv")

train_path='../input/train_images'

labels=train_data['diagnosis']

all_labels=to_categorical(labels)



images=[]

labels=[]

def generator_data():

    for  i,l in zip(train_data['id_code'],all_labels):

        image_data=cv2.imread(os.path.join(train_path,i+'.png'))

        image_data=cv2.resize(image_data,(224,224))

        image_data =image_data / 255.

        images.append(image_data)

        labels.append(l)

    return  images,labels 



train_data,train_labels=generator_data()

train_data=np.array(train_data)

train_labels=np.array(train_labels)



                

del images

del labels
from keras import layers

from keras  import models



from keras.layers import LeakyReLU

model=models.Sequential()

model.add(layers.Conv2D(64,(2,2),activation='relu',input_shape=(224,224,3)))

#model.add(LeakyReLU(alpha=0.05))

#model.add(layers.Dropout(0.2))

model.add(layers.MaxPooling2D(2,2))



model.add(layers.Conv2D(128,(2,2),activation='relu'))

#model.add(LeakyReLU(alpha=0.05))

#model.add(layers.Dropout(0.2))

model.add(layers.MaxPooling2D(2,2))



model.add(layers.Conv2D(128,(2,2),activation='relu'))

#model.add(LeakyReLU(alpha=0.05))

model.add(layers.MaxPooling2D(2,2))

#model.add(layers.Dropout(0.2))



model.add(layers.Flatten())

model.add(layers.Dense(512,activation='relu'))

#model.add(layers.Dropout(0.2))

model.add(layers.Dense(5,activation='softmax'))

model.summary()



from keras.callbacks import Callback, ModelCheckpoint

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])

history=model.fit(train_data,train_labels,batch_size=32,epochs=15,class_weight='auto',validation_split=0.15)
test_data=pd.read_csv('../input/test.csv')

test_path='../input/test_images'





target=[]

for  i in test_data['id_code']:

    test_image=cv2.imread(os.path.join(test_path,i+'.png'))

    test_image=cv2.resize(test_image,(224,224))

    test_image=test_image/255.

    test_image=test_image.reshape((1,)+test_image.shape)

    target.append(np.argmax(model.predict(test_image)))



target=np.array(target)

test_data['diagnosis']=target
print(test_data['diagnosis'])
test_data.to_csv('submission.csv',index=False)