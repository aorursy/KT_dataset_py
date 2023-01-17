# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras import applications

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras.preprocessing import image

from keras.models import Sequential,Model

from keras.layers import Dense,Flatten,Input

from keras.optimizers import SGD,RMSprop,Adagrad,Adadelta,Adam,Adamax,Nadam

from keras.losses import mean_squared_error

import seaborn as sns

import matplotlib.pyplot as plt

from keras.applications.imagenet_utils import preprocess_input

import cv2

from tqdm import tqdm_notebook as tqdm

from keras.models import load_model

import pickle

from sklearn.model_selection import train_test_split

import gc

from PIL import Image



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

%matplotlib inline

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import tensorflow as tf

def add_snow(img,factor):                      #Function to add snow

    image_HLS=cv2.cvtColor(img,cv2.COLOR_RGB2HLS)

    image_HLS=np.array(image_HLS,dtype=np.float64)

    brightness_coef=2.5

    snow_point=factor

    image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]= image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coef

    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255

    image_HLS=np.array(image_HLS,dtype=np.uint8)

    image_RGB=cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)

    return image_RGB



def add_brightness(img,factor):               #Function to change brightness

    image_HLS=cv2.cvtColor(img,cv2.COLOR_RGB2HLS)

    image_HLS=np.array(image_HLS,dtype=np.float64)

    image_HLS[:,:,1]=image_HLS[:,:,1]*factor

    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255

    image_HLS=np.array(image_HLS,dtype=np.uint8)

    image_RGB=cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)

    return image_RGB



def add_saturation(img,factor):           #Function to change saturation

    image_HLS=cv2.cvtColor(img,cv2.COLOR_RGB2HLS)

    image_HLS=np.array(image_HLS,dtype=np.float64)

    image_HLS[:,:,2]=image_HLS[:,:,2]*factor

    image_HLS[:,:,2][image_HLS[:,:,2]>255]  = 255

    image_HLS=np.array(image_HLS,dtype=np.uint8)

    image_RGB=cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)

    return image_RGB



def add_hue(img,factor):                 #Function to change hue

    image_HLS=cv2.cvtColor(img,cv2.COLOR_RGB2HLS)

    image_HLS=np.array(image_HLS,dtype=np.float64)

    image_HLS[:,:,0]=image_HLS[:,:,0]*factor

    image_HLS[:,:,0][image_HLS[:,:,0]>255]  = 255

    image_HLS=np.array(image_HLS,dtype=np.uint8)

    image_RGB=cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)

    return image_RGB



def generate_random_lines(imshape,slant,drop_length):

    drops=[]

    for i in range(1500): ## If You want heavy rain, try increasing this

        if slant<0:

            x= np.random.randint(slant,imshape[1])

        else:

            x= np.random.randint(0,imshape[1]-slant)

        y= np.random.randint(0,imshape[0]-drop_length)

        drops.append((x,y))

    return drops

        

    

def add_rain(image):

    

    imshape = image.shape

    slant_extreme=10

    slant= np.random.randint(-slant_extreme,slant_extreme) 

    drop_length=20

    drop_width=2

    drop_color=(200,200,200) ## a shade of gray

    rain_drops= generate_random_lines(imshape,slant,drop_length)

    

    for rain_drop in rain_drops:

        cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)

    image= cv2.blur(image,(7,7)) ## rainy view are blurry

    

    brightness_coefficient = 0.7 ## rainy days are usually shady 

    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS

    image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)

    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB

    return image_RGB



def flip(img):                #Function to return mirror image

    return np.fliplr(img)



def resize(image):                        #Function to resize the image

    image=image.resize((img_height,img_width),Image.ANTIALIAS)

    return image
def prepareImages(data, m, dataset):

    print("Preparing images")

    X_train = np.zeros((m, 100, 100, 3))

    count = 0

    

    for fig in data['Image']:

        #load images into images of size 100x100x3

        img = image.load_img("../input/humpback-whale-identification/"+dataset+"/"+fig, target_size=(100, 100, 3))

        x = image.img_to_array(img)

        x = preprocess_input(x)



        X_train[count] = x

        if (count%500 == 0):

            print("Processing image: ", count+1, ", ", fig)

        count += 1

    

    return X_train



train_data_dir='../input/humpback-whale-identification/train'

test_data_dir='../input/humpback-whale-identification/test'

img_height,img_width=100,100

train=pd.read_csv(train_data_dir+'.csv')

train_md=pd.get_dummies(train,columns=['Id'])

train_md.describe()


train_md=train_md[train_md.Id_new_whale!=1]

train_md.describe()
train_md.dropna()

train_md.drop(['Id_new_whale'],axis=1,inplace=True)

train_md.info()


y=train_md.drop(train_md.columns[0],axis=1)

names=list(y.columns.values)

y=np.array(y)

X=prepareImages(train_md,train_md.shape[0],"train")

X/=255

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.1,random_state=2)

X_train=X

y_train=y
print(X_train.shape)

print(X_val.shape)

print(y_train.shape)
model=applications.vgg16.VGG16(include_top=False,weights='imagenet',input_shape=(100,100,3),pooling='avg')

for layer in model.layers[:-5]:

    layer.trainable=False

print(model.output.shape)    
x=model.output

x=Dense(5004,activation='softmax')(x)

model_final=Model(inputs=model.input,outputs=x)

del X

del y

gc.collect()

print(model_final.output.shape)

model_final.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(),metrics=['acc'])

model_final.summary()
history=model_final.fit(X_train,y_train,epochs=20,verbose=1,batch_size=64,validation_data=(X_val,y_val))

gc.collect()
del X_train

gc.collect()
X_list=np.array(train['Image'])

print(X_list)
X=[]

for i in range(y_train.shape[0]):

   

    X.append(add_snow(cv2.resize(cv2.imread(train_data_dir+"/"+X_list[i]),(img_height,img_width)),50))

    

  

X=np.array(X)

X=X/255  

history=model_final.fit(X,y_train,epochs=16,verbose=1,batch_size=64,validation_data=(X_val,y_val))

del X

gc.collect()

X=[]

for i in range(y_train.shape[0]):

   

    X.append(add_snow(cv2.resize(cv2.imread(train_data_dir+"/"+X_list[i]),(img_height,img_width)),100))

    

X=X/255   

X=np.array(X)

history=model_final.fit(X,y_train,epochs=16,verbose=1,batch_size=64,validation_data=(X_val,y_val))

del X

gc.collect()
X=[]

for i in range(y_train.shape[0]):

   

    X.append(add_snow(cv2.resize(cv2.imread(train_data_dir+"/"+X_list[i]),(img_height,img_width)),150))

    

X=X/255    

X=np.array(X)

history=model_final.fit(X,y_train,epochs=16,verbose=1,batch_size=64,validation_data=(X_val,y_val))

del X

gc.collect()
X=[]

for i in range(y_train.shape[0]):

   

    X.append(add_brightness(cv2.resize(cv2.imread(train_data_dir+"/"+X_list[i]),(img_height,img_width)),0.5))

    



X=np.array(X)

X=X/255   

history=model_final.fit(X,y_train,epochs=16,verbose=1,batch_size=64,validation_data=(X_val,y_val))

del X

gc.collect()
X=[]

for i in range(y_train.shape[0]):

   

    X.append(add_brightness(cv2.resize(cv2.imread(train_data_dir+"/"+X_list[i]),(img_height,img_width)),1.25))

    



X=np.array(X)

X=X/255  

history=model_final.fit(X,y_train,epochs=16,verbose=1,batch_size=64,validation_data=(X_val,y_val))

del X

gc.collect()
X=[]

for i in range(y_train.shape[0]):

   

    X.append(add_brightness(cv2.resize(cv2.imread(train_data_dir+"/"+X_list[i]),(img_height,img_width)),1.75))

    

 

X=np.array(X)

X=X/255   

history=model_final.fit(X,y_train,epochs=16,verbose=1,batch_size=64,validation_data=(X_val,y_val))

del X

gc.collect()
X=[]

for i in range(y_train.shape[0]):

   

    X.append(add_saturation(cv2.resize(cv2.imread(train_data_dir+"/"+X_list[i]),(img_height,img_width)),0.5))

    



X=np.array(X)

X=X/255  

history=model_final.fit(X,y_train,epochs=16,verbose=1,batch_size=64,validation_data=(X_val,y_val))

del X

gc.collect()
X=[]

for i in range(y_train.shape[0]):

   

    X.append(add_saturation(cv2.resize(cv2.imread(train_data_dir+"/"+X_list[i]),(img_height,img_width)),1.5))

    



X=np.array(X)

X=X/255  

history=model_final.fit(X,y_train,epochs=16,verbose=1,batch_size=64,validation_data=(X_val,y_val))

del X

gc.collect()
X=[]

for i in range(y_train.shape[0]):

   

    X.append(add_hue(cv2.resize(cv2.imread(train_data_dir+"/"+X_list[i]),(img_height,img_width)),0.5))

    

 

X=np.array(X)

X=X/255   

history=model_final.fit(X,y_train,epochs=16,verbose=1,batch_size=64,validation_data=(X_val,y_val))

del X

gc.collect()
X=[]

for i in range(y_train.shape[0]):

   

    X.append(add_hue(cv2.resize(cv2.imread(train_data_dir+"/"+X_list[i]),(img_height,img_width)),1.5))

    



X=np.array(X)

X=X/255    

history=model_final.fit(X,y_train,epochs=16,verbose=1,batch_size=64,validation_data=(X_val,y_val))

del X

gc.collect()
X=[]

for i in range(y_train.shape[0]):

   

    X.append(add_rain(cv2.resize(cv2.imread(train_data_dir+"/"+X_list[i]),(img_height,img_width))))

    

   

X=np.array(X)

X=X/255 

history=model_final.fit(X,y_train,epochs=16,verbose=1,batch_size=64,validation_data=(X_val,y_val))

del X

gc.collect()
X=[]

for i in range(y_train.shape[0]):

   

    X.append(resize(flip(cv2.imread(train_data_dir+"/"+X_list[i]))))

    



X=np.array(X)

X=X/255    

history=model_final.fit(X,y_train,epochs=16,verbose=1,batch_size=64,validation_data=(X_val,y_val))

del X

gc.collect()
model_final.save_weights('model1weights.h5')

model_final.save('model1.h5')