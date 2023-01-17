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
#importing all package....

import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

import os

# Import Keras and other Deep Learning dependencies

from keras.models import Sequential

import time

from keras.optimizers import Adam

from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate

from keras.models import Model

import seaborn as sns

from keras.layers.normalization import BatchNormalization

from keras.layers.pooling import MaxPooling2D, AveragePooling2D

from keras.layers.merge import Concatenate

from keras.layers.core import Lambda, Flatten, Dense

from keras.initializers import glorot_uniform

from sklearn.preprocessing import LabelBinarizer

from keras.optimizers import *

from keras.engine.topology import Layer

from keras import backend as K

from keras.regularizers import l2

K.set_image_data_format('channels_last')

import cv2

import os

from skimage import io

import numpy as np

from numpy import genfromtxt

import pandas as pd

import tensorflow as tf





import numpy.random as rng

from sklearn.utils import shuffle

from keras.layers import *

from keras.models import Model

import cv2

from PIL import Image
#preparing train data...................



train_data=pd.read_csv("../input/train_relationships.csv")

test_data=pd.read_csv("../input/train_relationships.csv")



train_data[:5]
train_data.describe()

train_data.isnull().sum()



#add non-related column.......

list1=list(train_data.p1)



list2=list(train_data.p2)

new_list=list1[::-1]





temp_data=pd.DataFrame({'p1':new_list,'p2':list2})

#print(temp_data)

train_data=pd.concat([train_data,temp_data])

train_data.describe()



#creating whole label................................................

label=[]



for i in range(0,7196):

    if i<3599:

        label.append(1)

    else:

        label.append(0)

#print(label)
def add_image_path(x):

    image_path="../input/train/"+x

    temp_path="../input/train/F0002/MID1/P00017_face3.jpg"

    #print(image_path)

    if os.path.exists(image_path):

        #print(os.listdir(image_path)[0])

        path=os.path.join(image_path,os.listdir(image_path)[0])

        #print(path)

        return path

    else:

        return temp_path
train_data['p1_path']=train_data.p1.apply(lambda x:add_image_path(x))



train_data['p2_path']=train_data.p2.apply(lambda x:add_image_path(x))





train_data['is_related']=np.array(label)



from sklearn.utils import shuffle

train_data = shuffle(train_data)

#now check the train_data 

train_data.head()
train_data.fillna(method='ffill', axis=1)

train_data.isnull().sum()

train_data.tail(10)

train_data.isnull().sum()
#making x_train and y_train..here..........................



train_image_1=[]

train_image_2=[]

train_label=[]



h=0

w=0



for image_1,image_2,label in zip(train_data['p1_path'],train_data['p2_path'],train_data['is_related']):

    

        img_1=Image.open(image_1)

        #print(image_1,image_2)

        

        area=(50,52,150,160)

        cropped_1=img_1.crop(area)

        

        img_2=Image.open(image_2)

        cropped_2=img_2.crop(area)

        

        new1=np.array(cropped_1,dtype='uint8')/255

        new2=np.array(cropped_2,dtype='uint8')/255

        

        width, height = img_1.size

        if width>w:

            w=width

        if height>h:

            h=height

        

        train_image_1.append(new1)

        train_image_2.append(new2)

        

        train_label.append(label)





        

        

print(h,w)

train_image_1= np.array(train_image_1)

train_image_2= np.array(train_image_2)



"""

input_image_1 = Input(shape=(224, 224, 3))

input_image_2 = Input(shape=(224, 224, 3))



face = Conv2D(32, kernel_size=(3, 3))(input_image_1)

face = Conv2D(32, kernel_size=(3, 3))(face)

face = Flatten()(face)



sig = Conv2D(32, kernel_size=(3, 3))(input_image_2)

sig = Conv2D(32, kernel_size=(3, 3))(sig)

sig = Flatten()(sig)



output = concatenate([sig, face])

output = Dense(2, activation='softmax')(output)

model = Model(inputs=[input_face, input_sig], outputs=[output])

"""



#model-1 structure is defining here.........................



model1 = Sequential()

model1.add(Conv2D(64, (1,1), activation='relu', padding='same', input_shape=(108, 100, 3))) 

model1.add(MaxPooling2D((1, 1), padding='same'))

model1.add(Conv2D(64, (1, 1), activation='relu', padding='same'))

model1.add(MaxPooling2D((1, 1), padding='same'))

model1.add(Conv2D(32, (1, 1), activation='relu', padding='same'))

model1.add(MaxPooling2D((1, 1), padding='same'))

model1.add(Conv2D(32, (1, 1), activation='relu', padding='same'))

model1.add(MaxPooling2D((1, 1), padding='same'))





#model-2 structure is defining here..................

model2 = Sequential()

model2.add(Conv2D(64, (1, 1), activation='relu', padding='same', input_shape=(108, 100, 3))) 

model2.add(MaxPooling2D((1, 1), padding='same'))

model2.add(Conv2D(64, (1, 1), activation='relu', padding='same'))

model2.add(MaxPooling2D((1, 1), padding='same'))

model2.add(Conv2D(32, (1, 1), activation='relu', padding='same'))

model2.add(MaxPooling2D((1, 1), padding='same'))

model2.add(Conv2D(32, (1, 1), activation='relu', padding='same'))

model2.add(MaxPooling2D((1, 1), padding='same'))





#merging the two model................................

mergeOut=Add()([model1.output,model2.output])



mergeout=Flatten()(mergeOut)



mergeout=Dense(64,activation='relu')(mergeout)



mergeout=Dropout(.2)(mergeOut)

mergeout=Dense(32,activation='relu')(mergeout)

mergeout=Dropout(.2)(mergeout)



mergeout=Flatten()(mergeout)



#output layer.......

mergeout=Dense(1,activation='sigmoid')(mergeout)



newModel=Model([model1.input,model2.input],mergeout)



newModel.summary() 



newModel.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



val_image_1=train_image_1[:1000]

val_image_2=train_image_2[:1000]

val_label=train_label[:1000]
newModel.fit([train_image_1,train_image_2],train_label,epochs=100,validation_data=([val_image_1,val_image_2],val_label))



test_data=pd.read_csv("../input/sample_submission.csv")

test_data.describe()

test_data.head()
# new data frame with split value columns 

new = test_data["img_pair"].str.split("-", n = 1, expand = True) 

  

# making separate first name column from new data frame 

test_data["image_1"]= new[0] 

  

# making separate last name column from new data frame 

test_data["image_2"]= new[1] 
test_image_1=[]

test_image_2=[]

for img_1,img_2 in zip(test_data['image_1'],test_data['image_2']):

    img_1=os.path.join("../input/test/",img_1)

    img_2=os.path.join("../input/test/",img_2)

    img_1=Image.open(img_1)

        #print(image_1,image_2)

        

    area=(50,52,150,160)

    cropped_1=img_1.crop(area)

    img_2=Image.open(img_2)

    area=(50,52,150,160)

    cropped_2=img_2.crop(area)

        

    new1=np.array(cropped_1,dtype='uint8')

    new2=np.array(cropped_2,dtype='uint8')

    test_image_1.append(new1)

    test_image_2.append(new2)

    

        
plt.imshow(test_image_1[0])
pred=newModel.predict([test_image_1,test_image_2])

for i in range(0,100):

    print(pred[i])
subm=pd.DataFrame({'img_pair':test_data['img_pair']})

subm['is_related']=pred



subm.to_csv('submission.csv',index=False)
