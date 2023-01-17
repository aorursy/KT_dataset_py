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
!pip install efficientnet
!pip install tensorflow-addons
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpi

%matplotlib inline





from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_auc_score

import itertools



import efficientnet.keras as efn

import tensorflow_addons as tfa

from keras.models import Sequential,Model

from keras.layers import Dense,Dropout,Flatten,AvgPool2D,Activation,BatchNormalization,GlobalAveragePooling2D,GlobalMaxPooling2D,Conv2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

import keras
train = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")

test = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
train['image_id'] = train['image_id']+'.jpg'
train.shape
train.head()
test['image_id'] = test['image_id']+'.jpg'
x_train,val = train_test_split(train,test_size = 0.1)
x_train.head()
col = x_train.drop('image_id',axis=1).columns.tolist()

col
gen_data = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=12,width_shift_range=0.15,height_shift_range=0.1,zoom_range=0.2,rescale=1/255,fill_mode='nearest')
gen_train = gen_data.flow_from_dataframe(x_train,directory='/kaggle/input/plant-pathology-2020-fgvc7/images/',target_size=(300,300),x_col='image_id',y_col = col,class_mode='raw',shuffle = False,subset='training',batch_size=20)
val_gen = gen_data.flow_from_dataframe(val,directory='/kaggle/input/plant-pathology-2020-fgvc7/images/',target_size=(300,300),x_col = 'image_id',y_col = col,class_mode='raw',shuffle = False,batch_size=20)
gen_test = gen_data.flow_from_dataframe(test,directory='/kaggle/input/plant-pathology-2020-fgvc7/images/',target_size=(300,300),x_col='image_id',y_col=None,class_mode=None,shuffle=False,batch_size=20)
gen_train.next()[0].shape,gen_train.next()[1].shape
model = efn.EfficientNetB4(weights='imagenet',include_top=False,input_shape=(300,300,3))
x = model.output

x = GlobalAveragePooling2D()(x)

x = Dense(128,activation='relu')(x)

x = Dense(64,activation='relu')(x)

pred = Dense(4,activation='softmax')(x)
model = Model(inputs = model.input,outputs = pred)
optimizer = RMSprop()
model.compile(optimizer=optimizer,loss = tfa.losses.SigmoidFocalCrossEntropy(),metrics = ['accuracy'])
true = val.iloc[:,1::].values
batch_size = 20
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy',patience=3,verbose=1,factor=0.5,min_lr=0.0001)
result=model.fit_generator(gen_train,epochs = 5,steps_per_epoch=gen_train.n/batch_size,validation_data=val_gen,validation_steps=val_gen.n/batch_size,callbacks=[lr_reduction])
fig,ax = plt.subplots(2,1)



ax[0].plot(result.history['loss'],color = 'b',label = "Training loss")

ax[0].plot(result.history['val_loss'],color = 'r',label = "Validation Loss")

legend = ax[0].legend(loc='best',shadow = True)



ax[1].plot(result.history['accuracy'],color = 'b',label='Training Accuracy')

ax[1].plot(result.history['val_accuracy'],color='r',label = "Validation Accuracy")

legend = ax[1].legend(loc='best',shadow=True)
model.summary()
y_pred = model.predict_generator(val_gen,steps=val_gen.n/batch_size)

y_pred = y_pred.round().astype(int)
print(accuracy_score(true,y_pred))

print( )

print(f1_score(true,y_pred,average='macro'))

print( )

print(roc_auc_score(true,y_pred,average='macro'))
y_test = model.predict(gen_test,steps = gen_test.n/batch_size)
y_test.shape
sub=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')

sub.head()
len(sub),len(y_test)
for i,j in enumerate(['healthy','multiple_diseases','rust','scab']):

    sub[j]=y_test[:,i]
sub.head()
sub.to_csv("Submission.csv",index = False)