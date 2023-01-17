# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/shapes"))


# Any results you write to the current directory are saved as output.

import numpy as np
import os
import cv2
import pandas as pd
import joblib
from pathlib import Path
from keras.applications.vgg16 import preprocess_input
from keras.models import  Model

from keras.applications import vgg16
from keras.preprocessing import image
from keras.layers import Dense,Flatten,Dropout,InputLayer
from keras.models import Sequential

from keras import optimizers
from sklearn.cross_validation import train_test_split

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 19:24:29 2018

@author: ldhandu
"""


def load_images_from_folder(folder,lent):
    
    
    count = 0
    images = []
    
    
    for filename in os.listdir(folder):

#        img = cv2.imread(os.path.join(folder,filename))
        img = image.load_img(os.path.join(folder,filename),target_size=(224,224))
        img = image.img_to_array(img)
        img = preprocess_input(img)
        
        
        if img is not None:
            images.append(img)
        
        count = count + 1
        
        if count == lent:
            break
            
    return images

def array_to_df(arr_as_list,label_name):
    
    temp_arr = np.array(arr_as_list)
    temp_arr = np.reshape(temp_arr,(temp_arr.shape[0],224*224*3 ) )
    
    
        
    temp_label =[]
    
    for i in range(0,temp_arr.shape[0]):
        temp_label.append(label_name)
    
    temp_label  = np.asarray(temp_label)
    
    image_df = pd.DataFrame(temp_arr)
    label_df = pd.DataFrame(temp_label)    
        
    total_df = pd.concat([image_df,label_df],axis=1)
    
    return total_df

images_circle = load_images_from_folder('../input/shapes/circle',250)
images_square = load_images_from_folder('../input/shapes/square',250)
images_star = load_images_from_folder('../input/shapes/star',250)
images_triangle = load_images_from_folder('../input/shapes/triangle',250)



print(type(images_circle))

circle_df = array_to_df(images_circle,'circle')
square_df = array_to_df(images_square,'square')
star_df = array_to_df(images_star,'star')
triangle_df = array_to_df(images_triangle,'triangle')


total_df = pd.concat([circle_df,star_df,square_df,triangle_df],axis=0)
#print(total_df)



total_array = np.array(total_df)


x = total_array[:,0:-1]
y = total_array[:,-1]

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb.fit(y)
y = lb.transform(y)


print(x.shape)
print(y.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)

print('x_train : ', x_train.shape)
print('x_test : ', x_test.shape)
print('y_train : ', y_train.shape)
print('y_test : ', y_test.shape)
# vgg_model = vgg16.VGG16(weights='imagenet',input_shape=(224,224,3))

# my_model = Sequential()

# for i in vgg_model.layers[0:-1]:
#     my_model.add(i)

# for la in my_model.layers:
#     la.trainable = False
    
# my_model.add(Dense(4,activation='softmax'))

# my_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# my_model.summary()
# history = my_model.fit(np.reshape(x,(1000,224,224,3) ),y,epochs=3,verbose=1)
# n = my_model.get_layer(name='block1_conv1')
# with tf.Session()as sess:
#     sess.run(tf.global_variables_initializer())
#     a = np.asarray(n.weights)
#     print(a)
# pred_model = Sequential()
# pred_model.add(n)
# pred_model.predict(np.reshape(x[0],(224,224,3)))
# import matplotlib.pyplot as plt
# a= np.reshape(x[1],(224,224,3)).shape

# plt.imshow(a,cmap='Greys')
# # plt.imshow(vgg_model.get_weights()[0][:, :, :, 0].squeeze(), cmap='gray')
# top_layer = vgg_model.layers[1]
# print(top_layer.get_weights()[0][:,:,:,0].squeeze())
# # top_layer.get_weights()[0][:, :, :, 0]

# # top_layer = vgg_model.layers[4]
# # a =  np.asarray(top_layer.get_weights()) 
# # print(a[0].shape)
# # print(a[1].shape)

# #layer 3
# # # print(a[0][0][0][0].shape)
# # print(top_layer.get_weights()[0][:,:,:,0][:,:,1])
# # plt.imshow(top_layer.get_weights()[0][:,:,:,0][:,:,10].squeeze(), cmap='gray')

# #layer 4
# # print(top_layer.get_weights()[0][:,:,:,0][:,:,1].shape)
# # plt.imshow(top_layer.get_weights()[0][:,:,:,0][:,:,1], cmap='gray')

# # plt.imshow(top_layer.get_weights()[0][:,:,:,0].squeeze())

# w=10
# h=10
# fig=plt.figure(figsize=(75, 75))
# columns = 4
# rows = 5
# for i in range(1, columns*rows +1):
#     img = np.random.randint(10, size=(h,w))
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(img)
# plt.show()
# # vgg_model.outputs.index()
# a = my_model.layers
# x = a[0]
# x.output
# layer_outputs = [ layer.get_layer_ouput_at for layer in my_model.layers[:10]]
x_train = x_train/225
x_test = x_test/225
# vgg_model1 = vgg16.VGG16(weights='imagenet',input_shape=(224,224,3),include_top=False)

# # xt = Flatten()(vgg_model1.output)
# # xt = Dense(32, activation='relu' )(xt)
# # xt = Dropout(0.01)(xt)
# # xt = Dense(4, activation='softmax')(xt)

# # model = Model(inputs= vgg_model1.input, outputs=xt)
# #
# vgg_model1.summary()

# for i in vgg_model1.layers:
#     i.trainable = False


# # print(vgg_model1.output)

# output = vgg_model1.layers[-1].output 
# output = Flatten()(output) 
# vgg_model = Model(vgg_model1.input, output) 
# vgg_model.trainable = False 

# input_shape = vgg_model.output_shape[1]
# print(input_shape)
# model = Sequential()
# model.add(InputLayer(input_shape=(input_shape,))) 
# model.add(Dense(32, activation='relu',input_dim = input_shape) )
# model.add(Dropout(0.5))
# model.add(Dense(4, activation='softmax'))



# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# model.summary()
# model.get_config()
# history1 = model.fit(vgg_model1.predict(np.reshape(x_train,(750,224,224,3) ) ),y_train,epochs=10)
# #Feature Extraction Way

# model_vgg = vgg16.VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))

# features = model_vgg.predict(x_train.reshape(x_train.shape[0],224,224,3))


# transfer_learnt_model = Sequential()
# transfer_learnt_model.add(Flatten(input_shape=features.shape[1:]))
# transfer_learnt_model.add(Dense(32,activation='relu'))
# transfer_learnt_model.add(Dropout(0.5))
# transfer_learnt_model.add(Dense(4,activation='softmax'))


# transfer_learnt_model.compile(
        
#         loss = 'binary_crossentropy',
#         metrics=['acc'],
#         optimizer='adam'
#         )

# transfer_learnt_model.fit(features,y_train,epochs=10,shuffle=True)
#Feature Extraction way of transfer learning


vgg = vgg16.VGG16(include_top=False, weights='imagenet',input_shape=(224,224,3)) 
 
output = vgg.layers[-1].output 
output = Flatten()(output) 
vgg_model = Model(vgg.input, output) 
vgg_model.trainable = False 
 
for layer in vgg_model.layers: 
    layer.trainable = False 
 
vgg_model.summary() 


def get_bottleneck_features(model, input_imgs): 
    features = model.predict(input_imgs, verbose=0) 
    return features 

train_features_vgg = get_bottleneck_features(vgg_model, np.reshape(x_train,(750,224,224,3))) 
test_features_vgg = get_bottleneck_features(vgg_model, np.reshape(x_test,(250,224,224,3))) 

input_shape = vgg_model.output_shape[1] 
model = Sequential() 
model.add(InputLayer(input_shape=(input_shape,))) 
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3)) 
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.3)) 
model.add(Dense(4, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['accuracy']) 

model.summary()
history2 = model.fit(train_features_vgg,y_train,epochs=10,validation_data=(test_features_vgg,y_test))
