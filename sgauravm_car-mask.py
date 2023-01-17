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
from zipfile import ZipFile
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.backend import flatten
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D,UpSampling2D,Dropout,Input,MaxPooling2D,concatenate
from tensorflow.keras import Sequential,Model
from tensorflow.keras.optimizers import Adam
from keras.losses import binary_crossentropy, categorical_crossentropy
import tensorflow as tf
os.listdir('../input/carvana-image-masking-challenge/')
zip_file_name = '../input/carvana-image-masking-challenge/train_masks.zip'
with ZipFile(zip_file_name, 'r') as zip: 
    # printing all the contents of the zip file 
    zip.printdir() 
    # extracting all the files 
    print('Extracting all the files now...') 
    zip.extractall() 
    print('Done!')
train = os.listdir('./train/')
train_masks = os.listdir('./train_masks')
len(train)
os.listdir('./')
i=5
k = img_to_array(load_img('train/'+train[i]))
k_m = img_to_array(load_img('train_masks/'+train[i].split('.')[0] +'_mask.gif'))
fig,arr = plt.subplots(1,2)
fig.set_figheight(25)
fig.set_figwidth(30)
arr[0].imshow(k/255)
arr[1].imshow(k_m[:,:,2])
#Splitting the train and validation set
train_images,val_images = train_test_split(train,train_size=0.8)
len(train_images)
#Generator

def data_gen(dir_path_img,dir_path_mask,imgs,dims,batch_size):
    while True:
        idx = np.random.choice(np.arange(len(imgs)),batch_size)
        images =[]
        labels =[]
        for i in idx:
            img = Image.open(dir_path_img+imgs[i])
            images.append(np.array(img.resize(dims))/255)
            
            label = Image.open(dir_path_mask+imgs[i].split('.')[0]+'_mask.gif')
            labels.append(np.array(label.resize(dims)).reshape((dims)+(1,))/1.0)
        yield np.array(images),np.array(labels)
gen = data_gen('train/','train_masks/',train_images,(256,256),20)
img,lbl = next(gen)
img.shape,lbl.shape

#Testing generator
i=7
plt.imshow(img[i])
plt.imshow(lbl[i,:,:,0],alpha=0.5)
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)
res = next(gen)
res[1][1].shape
bce_dice_loss(res[1][0],res[1][1])
dice_coef(res[1][0],res[1][1])
img_dim1=512
img_dim2=512
model_cnn = Sequential()
model_cnn.add( Conv2D(16, 3, activation='relu', padding='same', input_shape=(img_dim1, img_dim2, 3)) )
model_cnn.add( Conv2D(32, 3, activation='relu', padding='same') )
model_cnn.add( Conv2D(1, 5, activation='sigmoid', padding='same') )
train_gen = data_gen('train/','train_masks/',train,(img_dim1,img_dim2),20)
model_cnn.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[dice_coef])
model_cnn.fit(train_gen, steps_per_epoch=100, epochs=10)
model_cnn.summary()
test_gen = data_gen('train/','train_masks/',val_images,(512,512),20)

res=model_cnn.predict(next(test_gen)[0])
res.shape
plt.imshow(res[0][:,:,0])
def UNET(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs, conv10)

    model.compile(optimizer = Adam(lr = 1e-3), loss = bce_dice_loss, metrics = ['accuracy',dice_coef])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
unet.summary()
dim1 = 256
dim2 = 256
unet = UNET(input_size=(dim1,dim2,3))
batch_size = 20
spe = len(train_images)//batch_size
gen_train = data_gen('train/','train_masks/',train_images,(dim1,dim2),batch_size)
gen_val = data_gen('train/','train_masks/',val_images,(dim1,dim2),batch_size)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
unet.fit(gen_train,validation_data=gen_val,steps_per_epoch=spe,epochs=15,validation_steps=len(val_images)//batch_size, callbacks=[early_stop]  )
unet.save('my_model.h5') 

batch_size = 20
gen_val = data_gen('train/','train_masks/',val_images,(dim1,dim2),batch_size)
val_in,val_true = next(gen_val)
val_pred = unet.predict(val_in)
i=19
plt.imshow(val_in[i])
plt.imshow(val_pred[i,:,:,0]>0.5,alpha=0.8)
dice_coef(val_true[0],(val_pred[0]).astype('double'))
val_true[0].shape
val_pred[0].shape
val_true[0][val_true[0]>0]
1/255
256*256