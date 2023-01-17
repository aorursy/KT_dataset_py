# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# Any results you write to the current directory are saved as output.
os.chdir(r'/kaggle/input/aptos2019-blindness-detection')

path=os.getcwd()

print(path)

print(os.listdir())
path_train=os.path.join(path,"train_images")

print("train_path: "+path_train)

path_test=os.path.join(path,"test_images")

print("train_path: "+path_test)

path_train_csv=os.path.join(path,"train.csv")

print("train_path: "+path_train_csv)

path_sample_csv=os.path.join(path,"sample_submission.csv")

print("train_path: "+path_sample_csv)

root_path = '/kaggle/input'

os.mkdir(os.path.join(root_path,"Train_new"))

os.chdir(r'/kaggle/input')

path=os.getcwd()

print(path)

print(os.listdir())

os.chdir(r'/kaggle/input/Train_new')

path_Train_new=os.getcwd()

print(path_Train_new)

                         
train=pd.read_csv(path_train_csv)

print(train.head(3))
a=train['diagnosis'].value_counts()

b=train['id_code'].value_counts()

print("Unique Id: "+str(len(a)))

print("total images: "+str(len(b)))

print("train_shape: "+str(train.shape))
train_images_name=train['id_code']

train_images_name=train_images_name.to_list()



train_name=[]

for name in train_images_name:

    train_name.append(name+'.png')
train_name=pd.DataFrame(train_name)

train=pd.concat([train,train_name],axis=1)
train.drop(['id_code'],axis=1,inplace=True)

train.columns=['diagnosis','id_code']
train[:3]
# filtering images w.r.t ID

label=train['diagnosis'].unique()

label_list=[]

for i in label:

    label_list.append(str(i))

print("Classes: "+str(len(label_list)))

d={}

for name in label_list:

    index=train['diagnosis']==int(name)

    a=train[index]

    d[name]=a['id_code'].tolist()
root_path = path_Train_new

os.mkdir(os.path.join(root_path,"0"))

os.mkdir(os.path.join(root_path,"1"))

os.mkdir(os.path.join(root_path,"2"))

os.mkdir(os.path.join(root_path,"3"))

os.mkdir(os.path.join(root_path,"4"))
print("Sub Directories of Train_New: "+str(os.listdir()))
os.chdir(path)

print(os.listdir())
# copying images to their respected ID's

os.chdir(path_train)

import shutil

for name in label_list:

    list=d[name]

    for f in list:

        path=os.path.join(path_Train_new,name)

        shutil.copy(f,path)

import keras

from keras.preprocessing.image import ImageDataGenerator

train_gen=ImageDataGenerator(rescale=1/255,rotation_range=40,width_shift_range=0.3,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,

                            fill_mode='nearest',validation_split=0.1)

train_data=train_gen.flow_from_directory(path_Train_new,subset='training',batch_size=50,target_size=(224,224))

test_data=train_gen.flow_from_directory(path_Train_new,subset='validation',batch_size=10,target_size=(224,224))
from keras.layers import Dense

from keras.models import load_model,Model

from keras import layers

n_classes=5

from keras.optimizers import RMSprop

in_layer = layers.Input((224,224,3))



conv1 = layers.Conv2D(96, 11, strides=4, activation='relu')(in_layer)

pool1 = layers.MaxPooling2D(3, 2)(conv1)

conv2 = layers.Conv2D(256, 5, strides=1, padding='same', activation='relu')(pool1)

pool2 = layers.MaxPooling2D(3, 2)(conv2)

conv3 = layers.Conv2D(384, 3, strides=1, padding='same', activation='relu')(pool2)

conv4 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu')(conv3)

pool3 = layers.MaxPooling2D(3, 2)(conv4)

flattened = layers.Flatten()(pool3)

dense1 = layers.Dense(4096, activation='relu')(flattened)

drop1 = layers.Dropout(0.5)(dense1)

dense2 = layers.Dense(4096, activation='relu')(drop1)

drop2 = layers.Dropout(0.5)(dense2)

preds = layers.Dense(n_classes, activation='softmax')(drop2)





model=Model(inputs=in_layer,outputs=preds)                                  # new model's summary

model.compile(loss="categorical_crossentropy", optimizer='RMSprop',metrics=["accuracy"])


import tensorflow as tf



class myCallback(tf.keras.callbacks.Callback):     # customized Callback class

    def on_epoch_end(self,epoch,logs={}):

        if(logs.get('val_accuracy')>0.70):

            print('cancelling since validation accuracy has been reached to 70%')

            self.model.stop_training=True

callbacks=myCallback()   
history=model.fit_generator(train_data,epochs=10,validation_data=test_data,callbacks=[callbacks])

#os.chdir(r'/kaggle/working')

#model_1.save("blind01.h5")
#os.chdir(r'/kaggle/input/blind01')

#os.listdir()

#from keras.models import load_model

#model=load_model("blind01.h5")
model.summary()
from PIL import Image

from numpy import asarray

test_images=os.listdir(path_test)

print(len(test_images))



y_pred=[]



for file in test_images:

    path_file=os.path.join(path_test,file)

    img=Image.open(path_file)

    img=asarray(img.resize((224,224)))

    img=img.reshape(1,224,224,3)

    y_pred.append(model.predict(img))
y_pred_np=np.array(y_pred)

print(y_pred_np.shape)

y_pred_np=y_pred_np.reshape(1928,5)

import pandas as pd

y_pred_final=y_pred_np.argmax(axis=1)
y_pred_final[10:20]

y_pred_df=pd.DataFrame(y_pred_final)

submission=pd.read_csv(path_sample_csv)

print(submission.head())

submission_n=pd.concat([submission,y_pred_df],axis=1)

submission_n.drop(['diagnosis'],axis=1,inplace=True)

submission_n.columns=['id_code','diagnosis']



os.chdir('/kaggle/working')

submission_n.to_csv("submission.csv",index=False)