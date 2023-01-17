# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# Any results you write to the current directory are saved as output.
os.chdir(r'/kaggle/input/humpback-whale-identification')

path=os.getcwd()

print(path)

print(os.listdir())
path_train=os.path.join(path,"train")

print("train_path: "+path_train)

path_test=os.path.join(path,"test")

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
a=train['Id'].value_counts()

b=train['Image'].value_counts()

print("Unique Id: "+str(len(a)))

print("total images: "+str(len(b)))

print("train_shape: "+str(train.shape))
import keras

from keras.preprocessing.image import ImageDataGenerator

# filtering images w.r.t ID

label=train['Id'].unique()

print("Classes: "+str(len(label)))

d={}

for name in label:

    index=train['Id']==name

    a=train[index]

    d[name]=a['Image'].tolist()

root_path = path_Train_new

for file in label:

    os.mkdir(os.path.join(root_path,file))
# copying images to their respected ID's

os.chdir(path_train)

import shutil

for name in label:

    list=d[name]

    for f in list:

        path=os.path.join(path_Train_new,name)

        shutil.copy(f,path)
import keras

from keras import applications

from keras.layers import Dense

from keras.models import Model

from keras.models import load_model

from keras import Sequential

from keras.layers.convolutional import Conv2D,MaxPooling2D

from keras.layers import Flatten
os.chdir(r'/kaggle/input/mobilenet')

mobile=load_model("MobileNet.h5")

mobile.summary()
x=mobile.layers[-6].output

p=Dense(5005,activation='softmax')(x)

model=Model(inputs=mobile.input,outputs=p)

model.summary()
for layer in model.layers[:-15]:

    layer.trainable=False
import os

from sklearn.metrics import confusion_matrix

from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self,epoch,logs={}):

        if(logs.get('val_accuracy')>0.90):

            print('cancelling since validation accuracy has been reached to 90%')

            self.model.stop_training=True

callbacks_3=myCallback()   

import tensorflow as tf

 

train_gen=ImageDataGenerator(rescale=1/255,rotation_range=40,width_shift_range=0.3,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2

                            ,fill_mode='nearest',validation_split=0.06)

train_data=train_gen.flow_from_directory(path_Train_new,target_size=(224,224),batch_size=50,subset='training')

validation_data=train_gen.flow_from_directory(path_Train_new,target_size=(224,224),batch_size=10,subset='validation')

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

model.fit_generator(train_data,epochs=10,validation_data=validation_data)





os.chdir('/kaggle/working')

model.save("Whale01.h5")

from keras.preprocessing import image

train=pd.read_csv(path_train_csv)

train_images=train['Image'].to_list()

    

y_pred=[]

for file in train_images:

    path_file=os.path.join(path_train,file)

    img=image.load_img(path_file,target_size=(224,224))

    img=image.img_to_array(img)

    img=np.expand_dims(img,axis=0)

    y_pred.append(model.predict(img))

y_pred_np=np.array(y_pred)

y_pred_np=y_pred_np.reshape(25361,5005)

y_pred_final=y_pred_np.argmax(axis=1)

train_labels=train['Id']

train_labels.replace({'Large':0,'Small':1},inplace=True)





         

            

  
test_images=os.listdir(path_test)

print(len(test_images))


y_pred_test=[]



for file in test_images:

    path_file=os.path.join(path_test,file)

    img=image.load_img(path_file,target_size=(224,224))

    img=image.img_to_array(img)

    img=np.expand_dims(img,axis=0)

    y_pred.append(model.predict(img))

y_pred_np=np.array(y_pred)

y_pred_np=y_pred_np.reshape(7960,5005)

y_pred_final=y_pred_np.argmax(axis=1)

train_labels=train['Id']

train_labels.replace({'Large':0,'Small':1},inplace=True)





         