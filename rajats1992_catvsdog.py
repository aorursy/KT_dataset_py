# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# from keras.preprocessing import image
# from os import walk
# data=[]
# input_file_names=[]
# #####get the file names of the images to read them one by one
# for (dirpath, dirnames, filenames) in walk("../input/dogs-vs-cats-redux-kernels-edition/train"):
#     input_file_names=filenames

# for x in input_file_names:
#     img_file_name=x##getting name of the image file
#     path=str("../input/train/"+img_file_name)####making proper path of the image file
#     i=image.load_img(path)####reading the image from the path 
#     i=i.resize((64,64))#####resizing the image 
#     iarray=image.img_to_array(i)####converting it to arrau
#     data.append(iarray)#####appending the image to the list

# plt.imshow(data[5])
# data=np.array(data)
# ####generating labels for the data 
# labels=[]
# for x in input_file_names:
#     if x.find("cat")>=0:
#         labels.append(0)
#     else:
#         labels.append(1)
# ###checking if the labels are properly tagged or not,both the classes have equal images 12500 each
# a=np.array(labels)
# np.unique(a,return_counts=True)

# ###reshaping the labels
# labels=a.reshape(25000,1)
# #####rescaling the data
# data=data/255.

from keras.layers import Conv2D,Dense,MaxPooling2D,BatchNormalization,Activation,Flatten
from keras import Sequential
from keras.initializers import glorot_normal
from keras import optimizers
from keras.models import Model
from keras.applications.imagenet_utils import decode_predictions
# model=Sequential()
# model.add(Conv2D(64,kernel_size=(2,2),strides=(1,1),kernel_initializer=glorot_normal(seed=100),input_shape=(64,64,3) ))
# model.add(Conv2D(64,kernel_size=(2,2),strides=(1,1),kernel_initializer=glorot_normal(seed=100)))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
# model.add(MaxPooling2D((2,2)))

# model.add(Conv2D(128,kernel_size=(2,2),strides=(1,1),kernel_initializer=glorot_normal(seed=100)))
# model.add(Conv2D(128,kernel_size=(2,2),strides=(1,1),kernel_initializer=glorot_normal(seed=100)))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
# model.add(MaxPooling2D((2,2)))

# model.add(Conv2D(256,kernel_size=(2,2),strides=(1,1),kernel_initializer=glorot_normal(seed=100)))
# model.add(Conv2D(256,kernel_size=(2,2),strides=(1,1),kernel_initializer=glorot_normal(seed=100)))
# model.add(Conv2D(256,kernel_size=(2,2),strides=(1,1),kernel_initializer=glorot_normal(seed=100)))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
# model.add(MaxPooling2D((2,2)))


# model.add(Conv2D(512,kernel_size=(2,2),strides=(1,1),kernel_initializer=glorot_normal(seed=100)))
# model.add(Conv2D(512,kernel_size=(2,2),strides=(1,1),kernel_initializer=glorot_normal(seed=100)))
# model.add(Conv2D(512,kernel_size=(2,2),strides=(1,1),kernel_initializer=glorot_normal(seed=100)))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
# model.add(MaxPooling2D((2,2)))


# model.add(Flatten())
# model.add(Dense(10,activation="relu",kernel_initializer=glorot_normal(seed=100)))
# model.add(Dense(1,activation="sigmoid",kernel_initializer=glorot_normal(seed=100)))








###train test split
# from sklearn.model_selection import train_test_split
# train_x,test_x,train_y,test_y=train_test_split(data,labels,test_size=0.2,random_state=100)


# train_y.shape
####compiling the model
# o=optimizers.adam()
# model.compile(loss="binary_crossentropy",metrics=["accuracy"],optimizer=o)



####fitting the model

# H=model.fit(train_x,train_y,epochs=16,validation_split=0.2)

# plt.plot(range(1,17),H.history["acc"])
# plt.plot(range(1,17),H.history["val_acc"])

#####making predictions on the test data
# preds=model.predict_classes(test_x)
# sum(preds==test_y)/len(test_y)
# test_data=[]
# input_test_file_names=[]
# #####get the file names of the images to read them one by one
# for (dirpath, dirnames, filenames) in walk("../input/test"):
#     input_test_file_names=filenames

# for x in input_test_file_names:
#     img_file_name=x##getting name of the image file
#     path=str("../input/test/"+img_file_name)####making proper path of the image file
#     i=image.load_img(path)####reading the image from the path 
#     i=i.resize((64,64))#####resizing the image 
#     iarray=image.img_to_array(i)####converting it to arrau
#     test_data.append(iarray)#####appending the image to the list
# test_data=np.array(test_data)
# test_data=test_data/255.
# test_preds=model.predict(test_data)
# test_preds=test_preds.reshape(len(test_preds))
####as per the submission file rule only numerical part from the file wwas needed
####like 3090.jpg should be saved in as 3090
# new_input_test_file_names=[]
# for x in input_test_file_names:
#     k=int(x[0:x.find(".jpg")])
#     new_input_test_file_names.append(k)
# df=pd.DataFrame({'id':new_input_test_file_names,
#              'label':test_preds})
# df.to_csv("submission.csv",index=False)
#################################TRYING OUT THE RESNET 50 ARCHITECTURE################################################
import random
l=[1,2,3]

# from keras.applications import resnet50
# from keras.preprocessing.image import ImageDataGenerator
# r=resnet50.ResNet50(weights='imagenet',include_top=False,input_shape=(197,197,3))
#########33getting data in in sahpe of (197,197,3) as min reqrmnt of resnet 50
# from keras.preprocessing import image
# from os import walk
# data=[]
# input_file_names=[]
# #####get the file names of the images to read them one by one
# for (dirpath, dirnames, filenames) in walk("../input/dogs-vs-cats-redux-kernels-edition/train/"):
#     input_file_names=filenames
    
# rand_imgs_indexes=random.sample(range(0, 24999), 14000)
# new_input_file_names=[]
# ######taking only 20000 random images
# for k in rand_imgs_indexes:
#     new_input_file_names.append(input_file_names[k])

# for x in new_input_file_names:
#     img_file_name=x##getting name of the image file
#     path=str("../input/dogs-vs-cats-redux-kernels-edition/train/"+img_file_name)####making proper path of the image file
#     i=image.load_img(path)####reading the image from the path 
#     i=i.resize((197,197))#####resizing the image 
#     iarray=image.img_to_array(i)####converting it to arrau
#     iarray=iarray/255.
#     data.append(iarray)#####appending the image to the list
# data=np.array(data)
# ####generating labels for the data 
# labels=[]
# for x in new_input_file_names:
#     if x.find("cat")>=0:
#         labels.append(0)
#     else:
#         labels.append(1)
# ###reshaping the labels
# a=np.array(labels)
# labels=a.reshape(14000,1)
# #########defining the new model by defining my own last layer
# new_model=r.output
# new_model=Flatten()(new_model)
# new_model=Dense(10)(new_model)
# new_model=Activation("relu")(new_model)
# new_model=Dense(1,activation="sigmoid")(new_model)

# final_model=Model(input=r.input,output=new_model)




###freezin all layers except from last 3 layers
# total_layers=len(final_model.layers)
# print(total_layers)
# for x in range(0,total_layers-4):
#     final_model.layers[x].trainable=False
    
     
# final_model.layers
##checking if the layers have been frozen or not
# for x in range(0,total_layers):
#     print(final_model.layers[x])
#     print(final_model.layers[x].trainable)
###train test split
# from sklearn.model_selection import train_test_split
# train_x,test_x,train_y,test_y=train_test_split(data,labels,test_size=0.2,random_state=100)

####compiling the model
# o=optimizers.adam()
# final_model.compile(loss="binary_crossentropy",metrics=["accuracy"],optimizer=o)
# final_model.fit(train_x,train_y,epochs=2,validation_split=0.2)
# predicted_test=final_model.predict(train_x)

##########################trying vgg19 model###################################
from keras.applications import VGG19
v=VGG19(weights="imagenet",include_top=False,input_shape=(120,120,3))
#########33getting data in in sahpe of (197,197,3) as min reqrmnt of resnet 50
import random
from keras.preprocessing import image
from os import walk
data=[]
input_file_names=[]
#####get the file names of the images to read them one by one
for (dirpath, dirnames, filenames) in walk("../input/dogs-vs-cats-redux-kernels-edition/train/"):
    input_file_names=filenames
    
rand_imgs_indexes=random.sample(range(0, 24999), 12000)
new_input_file_names=[]
######taking only 20000 random images
for k in rand_imgs_indexes:
    new_input_file_names.append(input_file_names[k])

for x in new_input_file_names:
    img_file_name=x##getting name of the image file
    path=str("../input/dogs-vs-cats-redux-kernels-edition/train/"+img_file_name)####making proper path of the image file
    i=image.load_img(path)####reading the image from the path 
    i=i.resize((120,120))#####resizing the image 
    iarray=image.img_to_array(i)####converting it to arrau
    iarray=iarray/255.
    data.append(iarray)#####appending the image to the list
data=np.array(data)
####generating labels for the data 
labels=[]
for x in new_input_file_names:
    if x.find("cat")>=0:
        labels.append(0)
    else:
        labels.append(1)
###reshaping the labels
a=np.array(labels)
labels=a.reshape(12000,1)
#########defining the new model by defining my own last layer
new_model=v.output
new_model=Flatten()(new_model)
new_model=Dense(10)(new_model)
new_model=Activation("relu")(new_model)
new_model=Dense(1,activation="sigmoid")(new_model)

final_model=Model(input=v.input,output=new_model)
###freezin all layers except from last 3 layers
total_layers=len(final_model.layers)
print(total_layers)
for x in range(0,total_layers-4):
    final_model.layers[x].trainable=False
    
     
# final_model.layers
##checking if the layers have been frozen or not
for x in range(0,total_layers):
    print(final_model.layers[x])
    print(final_model.layers[x].trainable)
###train test split
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(data,labels,test_size=0.2,random_state=100)

####compiling the model
o=optimizers.adam()
final_model.compile(loss="binary_crossentropy",metrics=["accuracy"],optimizer=o)
final_model.fit(train_x,train_y,batch_size=32,epochs=15,validation_split=0.2)
preds=final_model.predict(test_x) 
new_preds=[]
for x in preds:
    if x >0.5:
        new_preds.append(1)
    else:
        new_preds.append(0)
new_preds=np.array(new_preds)
new_preds=new_preds.reshape(len(new_preds),1)      
sum(new_preds==test_y)/len(test_y)
train_x=[]
test_x=[]
data=[]
labels=[]
# final_model.save_weights("vgg_19.h5")
########importing test file
data=[]
input_file_names=[]

#####get the file names of the images to read them one by one
for (dirpath, dirnames, filenames) in walk("../input/dogs-vs-cats-redux-kernels-edition/test/"):
    input_file_names=filenames

for x in input_file_names:
    img_file_name=x##getting name of the image file
    path=str("../input/dogs-vs-cats-redux-kernels-edition/test/"+img_file_name)####making proper path of the image file
    i=image.load_img(path)####reading the image from the path 
    i=i.resize((120,120))#####resizing the image 
    iarray=image.img_to_array(i)####converting it to arrau
    data.append(iarray)#####appending the image to the list
data=np.array(data)/255.
preds=final_model.predict(data)
preds=preds.reshape(len(preds))
####as per the submission file rule only numerical part from the file wwas needed
####like 3090.jpg should be saved in as 3090
new_input_test_file_names=[]
for x in input_file_names:
    k=int(x[0:x.find(".jpg")])
    new_input_test_file_names.append(k)
df=pd.DataFrame({'id':new_input_test_file_names,
             'label':preds})
df.to_csv("submission2.csv",index=False)