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
import numpy as np 

import tensorflow as tf

import pandas as pd

import os

from cv2 import imread, createCLAHE 

import cv2

from glob import glob

%matplotlib inline

import matplotlib.pyplot as plt
print(os.listdir("../input/Montgomery/MontgomerySet/"))

print(os.listdir("../input/Montgomery/MontgomerySet/ManualMask/"))
image_path=("../input/Montgomery/MontgomerySet/CXR_png/")

mask_path=("../input/Montgomery/MontgomerySet/ManualMask/")

right_mask_path=("../input/Montgomery/MontgomerySet/ManualMask/rightMask/")

left_mask_path=("../input/Montgomery/MontgomerySet/ManualMask/leftMask/")
 #获取数据索引

def get_data(image_path,right_mask_path,left_mask_path):

    image_path=os.listdir(image_path)

    right_mask_path=os.listdir(right_mask_path)

    left_mask_path=os.listdir(left_mask_path)

    

    print(len(image_path),len(right_mask_path),len(left_mask_path))

    

    data=list(set(image_path)&set(right_mask_path)&set(left_mask_path))

    print(len(data))

    return data

data=get_data(image_path,right_mask_path,left_mask_path)
#根据数据索引读取图片

def getMask(file_name):

    l = cv2.imread(left_mask_path+file_name)

    r = cv2.imread(right_mask_path+file_name)

    added_image = cv2.addWeighted(l,0.5,r,0.5,0)

    added_image=cv2.threshold(added_image,20,255,cv2.THRESH_BINARY)[1]  #大于20的值设置为255，其余设置为0

    return added_image[:,:,0]



def get_image(file_name):

    image=cv2.imread(image_path+file_name)

    return image[:,:,0]

#图片和标签全部一维化
#展示图片

file=data[15]

lung=get_image(file)

mask=getMask(file)

plt.figure(figsize=(15,8))

plt.subplot(121)

plt.imshow(lung)

plt.subplot(122)

plt.imshow(mask)



plt.show()
#将图片读入数组

from tqdm import tqdm

x_dim,y_dim=256,256

images=[cv2.resize(get_image(img),(x_dim,y_dim)) for img in tqdm(data)]

masks=[cv2.resize(getMask(img),(x_dim,y_dim)) for img in tqdm(data)]    #list形状
images=np.array(images).reshape(len(images),x_dim,y_dim,1)

masks=np.array(masks).reshape(len(masks),x_dim,y_dim,1)
images.shape,masks.shape
ls = np.random.randint(1,138,3)

for i in range(3):

    plt.figure(figsize=(15,10))

    plt.subplot(1,3,i+1)

    stacked = np.hstack((np.squeeze(images[ls[i]]),np.squeeze(masks[ls[i]])))

    plt.imshow(stacked)
from keras.optimizers import Adam

import keras.backend as K

from keras.losses import binary_crossentropy
reg_param = 1.0

lr = 6e-4

dice_bce_param = 0.0

use_dice = True



def dice_coef(y_true,y_pred,smooth=1):

    y_true_f=K.flatten(y_true)

    y_pred_f=K.flatten(y_pred)

    intersection=K.sum(y_true_f*y_pred_f)

    return(2.*intersection+smooth)/(smooth+K.sum(y_true_f)+K.sum(y_pred_f))





def dice_p_bce(in_gt, in_pred):

    return dice_bce_param*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)



def true_positive_rate(y_true, y_pred):

    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)

#建立u_net网络





import os



import numpy as np

import cv2

import matplotlib.pyplot as plt



from keras.models import *

from keras.layers import *

from keras.optimizers import *

from keras import backend as keras

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, LearningRateScheduler



from glob import glob

from tqdm import tqdm



def unet(input_size=(256,256,1)):

    inputs=Input(input_size)

    conv1=Conv2D(32,(3,3),activation='relu',padding='same')(inputs)

    conv1=Conv2D(32,(3,3),activation='relu',padding='same')(conv1) #[256,256,32]

    pool1=MaxPooling2D(pool_size=(2,2))(conv1)     #[128,128,32]

    

    conv2=Conv2D(64,(3,3),activation='relu',padding='same')(pool1)

    conv2=Conv2D(64,(3,3),activation='relu',padding='same')(conv2) #[128,128,64]

    pool2=MaxPooling2D(pool_size=(2,2))(conv2)    #[64,64,64]

    

    

    conv3=Conv2D(128,(3,3),activation='relu',padding='same')(pool2)

    conv3=Conv2D(128,(3,3),activation='relu',padding='same')(conv3)

    pool3=MaxPooling2D(pool_size=(2,2))(conv3)    #[32,32,128]

    

    

    conv4=Conv2D(256,(3,3),activation='relu',padding='same')(pool3)

    conv4=Conv2D(256,(3,3),activation='relu',padding='same')(conv4)

    pool4=MaxPooling2D(pool_size=(2,2))(conv4)    #[16,16,256]

    

    

    conv5=Conv2D(512,(3,3),activation='relu',padding='same')(pool4)

    conv5=Conv2D(512,(3,3),activation='relu',padding='same')(conv5)

                                                           #[16,16,512]

    

    up6=concatenate([Conv2DTranspose(256,(2,2),strides=(2,2),padding='same')(conv5),conv4],axis=3)  #[32,32,512]

    conv6=Conv2D(256,(3,3),activation='relu',padding='same')(up6)

    conv6=Conv2D(256,(3,3),activation='relu',padding='same')(conv6)    #[32,32,256]

    

    

    up7=concatenate([Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(conv6),conv3],axis=3) #[64,64,256]

    conv7=Conv2D(128,(3,3),activation='relu',padding='same')(up7)

    conv7=Conv2D(128,(3,3),activation='relu',padding='same')(conv7)    #[64,64,128]

    

    

    up8=concatenate([Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(conv7),conv2],axis=3)

    conv8=Conv2D(64,(3,3),activation='relu',padding='same')(up8)

    conv8=Conv2D(64,(3,3),activation='relu',padding='same')(conv8)    #[128,128,64]

    

    

    up9=concatenate([Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(conv8),conv1],axis=3)  #[256,256,64]

    conv9=Conv2D(32,(3,3),activation='relu',padding='same')(up9)

    conv9=Conv2D(32,(3,3),activation='relu',padding='same')(conv9)    #[256,256,32]

    

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    

    return Model(inputs=inputs,outputs=[conv10])

    

    
net=unet()

net.summary()
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

weight_path="{}_weights.best.hdf5".format('cxr_reg')



checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)



reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 

                                   patience=3, 

                                   verbose=1, mode='min', epsilon=0.0001, cooldown=2, min_lr=1e-6)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=10) # probably needs to be more patient, but kaggle time is limited

callbacks_list = [checkpoint, early, reduceLROnPlat]




from IPython.display import clear_output

from keras.optimizers import Adam 

from sklearn.model_selection import train_test_split



net.compile(optimizer=Adam(lr=lr), 

              loss=[dice_p_bce], 

           metrics = [true_positive_rate, 'binary_accuracy'])



train_vol, test_vol, train_seg, test_seg = train_test_split((images-127.0)/127.0, 

                                                            (masks>127).astype(np.float32), 

                                                            test_size = 0.2, 

                                                            random_state = 2018)

loss_history = net.fit(x = train_vol,

                       y = train_seg,

                       

                  epochs = 100,

                  validation_data =(test_vol,test_seg) ,

                  callbacks=callbacks_list

                 )



fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))

ax1.plot(loss_history.history['loss'], '-', label = 'Loss')

ax1.plot(loss_history.history['val_loss'], '-', label = 'Validation Loss')

ax1.legend()



ax2.plot(100*np.array(loss_history.history['binary_accuracy']), '-', 

         label = 'Accuracy')

ax2.plot(100*np.array(loss_history.history['val_binary_accuracy']), '-',

         label = 'Validation Accuracy')

ax2.legend()
prediction = net.predict(images[0:10])

plt.figure(figsize=(20,10))



for i in range(0,6,3):

    plt.subplot(2,3,i+1)

    plt.imshow(np.squeeze(images[i]))

    plt.xlabel("Base Image")

    

    plt.subplot(2,3,i+2)

    plt.imshow(np.squeeze(prediction[i]))

    plt.xlabel("Pridiction")

    plt.subplot(2,3,i+3)

    plt.imshow(np.squeeze(masks[i]))

    plt.xlabel("Segmentation Ground Truth")