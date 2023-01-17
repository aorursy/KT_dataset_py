# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import matplotlib.pyplot as plt

import seaborn as sns



import tensorflow as tf

from sklearn.metrics import confusion_matrix

from keras.preprocessing import image

from keras import models

from keras import layers

from keras import optimizers

from keras import applications

from keras.optimizers import Adam

from keras.models import Sequential, Model 

from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D,Conv2D,MaxPooling2D,BatchNormalization

from keras import backend as k 

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping



import keras

from keras.optimizers import SGD

from sklearn.model_selection import train_test_split



#print(os.listdir("../input/all/All"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/all/All/GTruth.csv')

df.head()
#Number of classes

df['Ground_Truth'].value_counts()
kv_dict= dict(zip(df['Id'].values,df['Ground_Truth'].values))

kv_dict
#Reading Image data and converting it into pixels and separating class labels

Data=[]

Label=[]

directory='../input/all/All'



for filename in os.listdir(directory) :

    if filename.endswith(".jpeg") or filename.endswith(".jpg"):

        

        Label.append(kv_dict.get(int(filename.split('.')[0])))

        filename=os.path.join(directory, filename)

        im=image.load_img(filename,target_size=(224, 224))

        im=np.reshape(im,(224,224,3))

        im=im.astype('float32') / 255

        Data.append(im)

    else:

        continue    
#Train Test Split

X_train, X_1, y_train, y_1 = train_test_split(np.array(Data), np.array(Label), test_size=0.2, random_state=42,stratify=Label)



#Train Test Split

X_cv, X_test, y_cv, y_test = train_test_split(X_1, y_1, test_size=0.2, random_state=42,stratify=y_1)
X_train.shape
X_cv.shape
X_test.shape
img_width=224

img_height=224
from keras import backend as K



if K.image_data_format() == 'channels_first':

    input_shape = (3, img_width, img_height)

    X_train=X_train.reshape(X_train.shape[0],3,img_width,img_height)

    X_cv=X_cv.reshape(X_cv.shape[0],3,img_width,img_height)

    X_test=X_test.reshape(X_test.shape[0],3,img_width,img_height)

    

else:

    input_shape = (img_width, img_height, 3)

    X_train=X_train.reshape(X_train.shape[0],img_width,img_height,3)

    X_cv=X_cv.reshape(X_cv.shape[0],img_width,img_height,3)

    X_test=X_test.reshape(X_test.shape[0],img_width,img_height,3)

    
del Data
#Function to Plott train and Test loss



def plt_dynamic(x,vy,ty,ax,colors=['b']):

  ax.plot(x,vy,'b',label='Validation Loss')

  ax.plot(x,ty,'r',label='Train Loss')

  plt.legend()

  plt.grid()

  fig.canvas.draw()
#Variables defined

epoch=10

batch=32

num_classes=1
#Model Defining

model=Sequential()



model.add(Conv2D(32,kernel_size=(3,3),

                activation='relu',

                input_shape=input_shape,

                kernel_initializer='he_normal'))  

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),

                activation='relu',

                kernel_initializer='he_normal'))

model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(64,kernel_size=(3,3),

                activation='relu',

                kernel_initializer='he_normal'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512,activation='relu',kernel_initializer='he_normal'))

model.add(Dropout(0.25))

model.add(Dense(256,activation='relu',kernel_initializer='he_normal'))

model.add(Dropout(0.4))

model.add(Dense(128,activation='relu',kernel_initializer='he_normal'))

model.add(Dropout(0.3))

model.add(BatchNormalization())

model.add(Dense(64,activation='relu',kernel_initializer='he_normal'))

model.add(Dropout(0.5))

model.add(Dense(num_classes,activation='sigmoid',kernel_initializer='glorot_normal'))

model.summary()
#Model Compile

model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])



# Train

#training = model.fit_generator(generator.flow(X_train,y_train, batch_size=batch)

                              #,epochs=epoch

                               # ,validation_data=[X_cv, y_cv]

                                #,steps_per_epoch=10,verbose=1)

his=model.fit(X_train,y_train,batch_size=batch,epochs=epoch,verbose=1,validation_data=(X_cv,y_cv))



#Plotting Train and Validation Loss

fig,ax=plt.subplots(1,1)

ax.set_xlabel('Epochs')

ax.set_ylabel('Binary Cross Entropy')



x=list(range(1,epoch+1))



vy=his.history['val_loss']

ty=his.history['loss']

plt_dynamic(x,vy,ty,ax)
#Test Accuracy

score=model.evaluate(X_test,y_test,verbose=0)

print("The test accuracy for the model is %f "%(score[1]*100))
y_pred=model.predict(X_test).round()
#Plotting Confusion Matrix

x=confusion_matrix(y_test,y_pred)

Cm_df=pd.DataFrame(x,index=['Pneumonia',' No Pneumonia'],columns=['Pneumonia','No Pneumonia'])



sns.set(font_scale=1.5,color_codes=True,palette='deep')

sns.heatmap(Cm_df,annot=True,annot_kws={'size':16},fmt='d',cmap='YlGnBu')

plt.ylabel("True Label")

plt.xlabel("Predicted Label")

plt.title('Confusion Matrix')