import glob

import numpy as np

import pandas as pd 

import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img, img_to_array
def preprocessing(path,output): 

    X=[]

    Y=[]

    path_of_data=glob.glob(path+'/*.jpg') 

    for i in path_of_data :

            image=load_img(i, target_size=(150,150)) 

            image=img_to_array(image) 

           # image=image/255.0 

            X.append(image) 

            Y.append(output) 

    return np.array(X),np.array(Y) 
#paths=glob.glob(r"/content/seg_test/seg_test/*")

#l=len(r"/content/seg_test/seg_test/*")

#labels=[]

#for path in paths:

#    labels.append(path[l:])

#    print(labels)
label=[0,1,2,3,4,5]

X_building, Y_building  = preprocessing(r"../input/intel-image-classification/seg_train/seg_train/buildings",label[0])

X_forest,Y_forest  = preprocessing(r"../input/intel-image-classification/seg_train/seg_train/forest",label[1])

X_glacier,Y_glacier  = preprocessing(r"../input/intel-image-classification/seg_train/seg_train/glacier",label[2])

X_mount,Y_mount  = preprocessing(r"../input/intel-image-classification/seg_train/seg_train/mountain",label[3])

X_sea,Y_sea  = preprocessing(r"../input/intel-image-classification/seg_train/seg_train/sea",label[4])

X_street,Y_street  = preprocessing(r"../input/intel-image-classification/seg_train/seg_train/street",label[5])



X=np.concatenate((X_building,X_forest,X_glacier,X_mount,X_sea,X_street),axis=0) 

Y=np.concatenate((Y_building,Y_forest,Y_glacier,Y_mount,Y_sea,Y_street),axis=0) 
X.shape
Y.shape
from keras.applications.vgg16 import VGG16 

vgg_model = VGG16(input_shape=[150,150,3], weights='imagenet', include_top=False)
for layer in vgg_model.layers: # here we are iterating all the layer

  layer.trainable = False 

from keras.layers import Flatten,Dense

x = Flatten()(vgg_model.output)

prediction = Dense(6, activation='softmax')(x) # this statement add categorize in last layer of x. using activation function softmax. using this statement , our mannual create output layer append in vgg16 model layer



from keras.models import Model



model = Model(vgg_model.input, prediction) # give input is vgg16 put and outputs is prediction to our vgg16 model



#model_vgg = Model(pretrained_model.input, x) 
model.compile(

  loss='sparse_categorical_crossentropy',

  optimizer='adam',

  metrics=['accuracy']

)
model.fit(X,Y,epochs=10,verbose=1)

# model_vgg.fit(x_train,y_train,epochs=1,validation_data=(x_test,y_test))


