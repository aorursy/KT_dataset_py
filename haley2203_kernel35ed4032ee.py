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
#setup

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import ast

import json

from PIL import Image, ImageDraw 

test_simplified = pd.read_csv("../input/test_simplified.csv")

test_simplified.head()
display_test=pd.DataFrame()

display_test=display_test.append(pd.read_csv("../input/test_simplified.csv",usecols=['drawing'],nrows=50))

display_test.head()
display_test['drawing'] = display_test['drawing'].apply(json.loads)

display_test.shape
figrows=10

figcols=5

fig, axs = plt.subplots(nrows=figrows, ncols=figcols, sharex=True, sharey=True, figsize=(16, 10))

for i, drawing in enumerate(display_test.drawing):

    ax = axs[i // figcols, i % figcols]

    for x, y in drawing:

        ax.plot(x, -np.array(y), lw=3)

    ax.axis('off')

plt.show()
sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission.head()
path_train = '../input/train_simplified/'

train0 = pd.read_csv(path_train+(os.listdir(path_train)[0]))

train0.head()
path_train = '../input/train_simplified/'

display_samples=pd.DataFrame()

display_samples=display_samples.append(pd.read_csv(path_train+(os.listdir(path_train)[0]),usecols=['drawing', 'word'],nrows=50))

display_samples.head()
display_samples['drawing'] = display_samples['drawing'].apply(json.loads)

display_samples.shape
figrows=10

figcols=5

fig, axs = plt.subplots(nrows=figrows, ncols=figcols, sharex=True, sharey=True, figsize=(16, 10))

for i, drawing in enumerate(display_samples.drawing):

    ax = axs[i // figcols, i % figcols]

    for x, y in drawing:

        ax.set_title(display_samples.word.iloc[i])

        ax.plot(x, -np.array(y), lw=3)

    ax.axis('off')

plt.show()
len(os.listdir(path_train))
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import ast

import os

from glob import glob

from tqdm import tqdm

from dask import bag

import cv2

import tensorflow as tf

from tensorflow import keras

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.metrics import top_k_categorical_accuracy

from keras.metrics import sparse_top_k_categorical_accuracy

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
classfiles=os.listdir('../input/train_simplified/')

numstonames={i : v[:-4].replace(' ','_') for i , v in enumerate(classfiles)}



num_class=340

imheight,imwidth=32,32

ims_per_class=2000
def stroke_to_img(strokes):

    img=np.zeros((256,256))

    for each in ast.literal_eval(strokes):

        for i in range(len(each[0])-1):

            cv2.line(img,(each[0][i],each[1][i]),(each[0][i+1],each[1][i+1]),255,5)

    img=cv2.resize(img,(32,32))

    img=img/255

    return img
rd=np.random.randint(340)

ranclass=numstonames[rd]

ranclass=ranclass.replace('_',' ')

rdpath='../input/train_simplified/'+ranclass+'.csv'

one=pd.read_csv(rdpath,usecols=['drawing','recognized','word'],nrows=10)

one=one[one.recognized==True].head(2)

name=one['word'].head(1)

strk=one['drawing']

pic=[]

for s in strk:

    pic.append(stroke_to_img(s))

name=name.values
train_grand=[]

class_paths = glob('../input/train_simplified/*.csv')

for i , c in enumerate(tqdm(class_paths[0:num_class])):

    train=pd.read_csv(c,usecols=['drawing','recognized'],nrows=ims_per_class*2)

    train=train[train.recognized==True].head(ims_per_class)

    imagebag=bag.from_sequence(train.drawing.values).map(stroke_to_img)

    trainarray=np.array(imagebag.compute())

    trainarray=np.reshape(trainarray,(ims_per_class,-1))

    labelarray=np.full((train.shape[0],1),i)

    trainarray=np.concatenate((labelarray,trainarray),axis=1)

    train_grand.append(trainarray)



train_grand=np.array([train_grand.pop() for i in np.arange(num_class)])

train_grand=train_grand.reshape((-1,(imheight*imwidth+1)))



del trainarray

del train
valfrac=0.2

cutpt=int(valfrac*train_grand.shape[0])



np.random.shuffle(train_grand)

y_train, x_train=train_grand[cutpt:,0],train_grand[cutpt:,1:]

y_val,x_val=train_grand[0:cutpt,0], train_grand[0:cutpt,1:]



del train_grand



x_train=x_train.reshape(x_train.shape[0],imheight,imwidth,1)

x_val=x_val.reshape(x_val.shape[0],imheight,imwidth,1)
model =Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding='same',activation='relu',input_shape=(imheight,imwidth,1)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(680,activation='relu'))

model.add(Dense(num_class,activation='softmax'))

model.summary()
def top_3_accuracy(x,y):

    t3=sparse_top_k_categorical_accuracy(x,y,3)

    return t3



reduceLROnPlat=ReduceLROnPlateau(monitor='val_loss',factor=0.3,patience=5,verbose=1,mode='auto',min_delta=0.005,cooldown=5,min_lr=0.001)

earlystop=EarlyStopping(monitor='val_acc',mode='max',patience=5)

callbacks=[reduceLROnPlat,earlystop]



model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy',top_3_accuracy])



history=model.fit(x=x_train,y=y_train,batch_size=150,epochs=500,validation_data=(x_val,y_val),callbacks=callbacks,verbose=1)
acc=history.history['acc']

val_acc=history.history['val_acc']

loss= history.history['loss']

val_loss=history.history['val_loss']



epochs=range(1,len(acc)+1)



plt.plot(epochs,acc,label='Training acc')

plt.plot(epochs,val_acc,label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs,loss,label='Training loss')

plt.plot(epochs,val_loss,label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
ttvlist=[]

reader=pd.read_csv('../input/test_simplified.csv',index_col=['key_id'],chunksize=2048)

for chunk in tqdm(reader,total=55):

    imagebag=bag.from_sequence(chunk.drawing.values).map(stroke_to_img)

    testarray=np.array(imagebag.compute())

    testarray=np.reshape(testarray,(testarray.shape[0],imheight,imwidth,1))

    testpreds=model.predict(testarray,verbose=0)

    ttvs=np.argsort(-testpreds)[:,0:3]

    ttvlist.append(ttvs)

ttvarray=np.concatenate(ttvlist)

pred_df=pd.DataFrame({'first': ttvarray[:,0],'second':ttvarray[:,1],'third':ttvarray[:,2]})

pred_df=pred_df.replace(numstonames)

pred_df['words']=pred_df['first']+' '+pred_df['second']+' '+pred_df['third']



sub=pd.read_csv('../input/sample_submission.csv',index_col=['key_id'])

sub['word']=pred_df.words.values

sub.to_csv('submission_summer.csv')