import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import cv2

from tqdm import tqdm

import tensorflow as tf

from sklearn.model_selection import train_test_split

from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Dropout,Concatenate,Conv2DTranspose

from keras.utils import Sequence

from sklearn.utils import shuffle

from keras.models import Model

from keras.optimizers import Adam

import keras.applications as KA

from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,TensorBoard

from albumentations import (

    PadIfNeeded,

    HorizontalFlip,

    VerticalFlip,    

    CenterCrop,    

    Crop,

    Compose,

    Transpose,

    RandomRotate90,

    ElasticTransform,

    GridDistortion, 

    OpticalDistortion,

    RandomSizedCrop,

    OneOf,

    CLAHE,

    RandomBrightnessContrast,    

    RandomGamma    

)
train_df_path='../input/severstal-steel-defect-detection/train.csv'

test_df_path='../input/severstal-steel-defect-detection/sample_submission.csv'

train_img_path='../input/severstal-steel-defect-detection/train_images/'

test_img_path='../input/severstal-steel-defect-detection/test_images/'

model_path='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5'
BATCH_SIZE=32

HEIGHT,WIDTH=128,128

EPOCHS=8
#Train data

train_df=pd.read_csv(train_df_path)

train_df.head()
train_df['ImageId_ClassId']=[train_img_path+ix for ix in train_df['ImageId_ClassId']]

#Fill Empty Encoding with 0

train_df['EncodedPixels'].fillna(0,inplace=True)
#Create list of list containing image index with their respective encoding

train_data=[]

for ix in tqdm(range(0,train_df.shape[0],4)):

    tmp=[]

    tmp.append(train_df.loc[ix,'ImageId_ClassId'].split('_')[0]+'_'+train_df.loc[ix,'ImageId_ClassId'].split('_')[1])

    for j in range(ix,ix+4):

        tmp.append(train_df.loc[j,'EncodedPixels'])

    train_data.append(tmp)
def rleToMask(rleString,height,width,h,w,resize=False):

    rows,cols = height,width

    rleNumbers = [int(numstring) for numstring in rleString.split(' ')]

    rlePairs = np.array(rleNumbers).reshape(-1,2)

    img = np.zeros(rows*cols,dtype=np.float32)

    for index,length in rlePairs:

        index -= 1

        img[index:index+length] = 1.0

    img = img.reshape(cols,rows)

    img = img.T

    if resize:

        img=cv2.resize(img,(h,w))

    return img
#Image Plotting Along with masks

class_color=['Reds','Blues','Greens','Oranges']

h,w=256,1600

fig=plt.figure(figsize=(12,12))

rows,cols=6,1

for i in range(1,rows*cols+1):

    img=cv2.imread(train_data[i-1][0])

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    fig.add_subplot(rows,cols,i)

    plt.imshow(img)

    for j in range(4):

        msk_encode=train_data[i-1][j+1]

        if msk_encode==0:

            continue

        else:

            mask=rleToMask(msk_encode,256,1600,256,256)

            plt.imshow(img,cmap='gray')

            plt.imshow(mask,cmap=class_color[j],alpha=0.2)            

plt.show()
train,val=train_test_split(train_data,test_size=0.15,random_state=13)

print('Train Size: {}'.format(len(train)))

print('Val Size: {}'.format(len(val)))
def sep_indexes(indexes_):

    img_tmp=[]

    mask_tmp=[]

    for ix in indexes_:

        img_tmp.append(ix[0])

        mask_tmp.append(ix[1:])

    return img_tmp,mask_tmp
def aug_fx(image,mask):

    aug = PadIfNeeded(p=1, min_height=HEIGHT, min_width=WIDTH)

    augmented = aug(image=image, mask=mask)

    

    aug = CenterCrop(p=1, height=HEIGHT, width=WIDTH)

    augmented = aug(image=augmented['image'], mask=augmented['mask'])

    

    aug = HorizontalFlip(p=1)

    augmented = aug(image=augmented['image'], mask=augmented['mask'])

    

    aug = VerticalFlip(p=1)

    augmented = aug(image=augmented['image'], mask=augmented['mask'])

    

    aug = Transpose(p=1)

    augmented = aug(image=augmented['image'], mask=augmented['mask'])

    

    aug = RandomRotate90(p=1)

    augmented = aug(image=augmented['image'], mask=augmented['mask'])

    

    aug = GridDistortion(p=1)

    augmented = aug(image=augmented['image'], mask=augmented['mask'])    

    

    return augmented['image'],augmented['mask']

class customGenerator(Sequence):

    def __init__(self,data_list,batch_size,height,width,is_train=True):

        self.indexes,self.mask_ids=sep_indexes(data_list)

        self.batch_size=batch_size

        self.height=height

        self.width=width

        self.is_train=is_train

    

    def __len__(self):

        return int(np.ceil(len(self.indexes)/self.batch_size))

    

    def __getitem__(self,idx):

        batch_x=self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_y=self.mask_ids[idx*self.batch_size:(idx+1)*self.batch_size]

        if self.is_train:

            return self.train_generator(batch_x,batch_y)

        else:

            return self.val_generator(batch_x,batch_y)

    

    def on_epoch_end(self):

        if(self.is_train):

            self.indexes,self.mask_ids = shuffle(self.indexes,self.mask_ids)

        else:

            pass

    

    def load_images(self,img_ids):

        tmp=np.zeros((len(img_ids),self.height,self.width,3))

        for ix,id_ in enumerate(img_ids):

            img=cv2.imread(id_)

            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            img=img.astype(np.float32) / 255.

            img=cv2.resize(img,(self.height,self.width))

            #img=np.expand_dims(img,-1)

            tmp[ix]=img

        return tmp

    

    def load_masks(self,mask_ids_):

        tmp=np.zeros((len(mask_ids_),self.height,self.width,4))

        for ix,enc in enumerate(mask_ids_):

            for j,enc_ in enumerate(enc):

                if enc_==0:

                    continue

                else:

                    mask=rleToMask(enc_,256,1600,self.height,self.width,resize=True)

                    tmp[ix,:,:,j]=mask

        return tmp

    

    def train_generator(self,batch_x,batch_y):

        image_batch=self.load_images(batch_x)

        mask_batch=self.load_masks(batch_y)

        

        #Augmentation

        for ix in range(len(image_batch)):

            image_batch[ix],mask_batch[ix]=aug_fx(image_batch[ix],mask_batch[ix])

            

        return image_batch,mask_batch

    

    def val_generator(self,batch_x,batch_y):

        image_batch=self.load_images(batch_x)

        mask_batch=self.load_masks(batch_y)

        return image_batch,mask_batch
train_gen=customGenerator(train,BATCH_SIZE,HEIGHT,WIDTH)

val_gen=customGenerator(val,BATCH_SIZE,HEIGHT,WIDTH,is_train=False)
from keras.applications.densenet import DenseNet121

base_model=DenseNet121(weights=model_path,input_shape=(128,128,3),include_top=False)

x=UpSampling2D(16)(base_model.output)

x=Conv2D(64,(3,3),strides=1,activation='relu',padding='same')(x)

x=UpSampling2D(2)(x)

x=Conv2D(32,(3,3),strides=1,activation='relu',padding='same')(x)

out = Conv2D(4, 1, activation = 'sigmoid')(x)

model=Model(base_model.input,out)

model.summary()
from keras import backend as K



def dice_coef(y_true, y_pred, smooth=1):

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)

    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)



def dice_coef_loss(y_true, y_pred):

    return 1-dice_coef(y_true, y_pred)
model.compile(loss='binary_crossentropy',optimizer=Adam(0.0001),metrics=[dice_coef])
train_steps=int(np.ceil(len(train)/BATCH_SIZE))

val_steps=int(np.ceil(len(val)/BATCH_SIZE))
mc=ModelCheckpoint('seg_model.h5',monitor='val_loss',mode='min',save_best_only=True,period=1,verbose=1)

rop=ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=2,min_lr=0.0000001)
history=model.fit_generator(train_gen,epochs=EPOCHS,steps_per_epoch=train_steps,

                    validation_data=val_gen,validation_steps=val_steps,use_multiprocessing=True,callbacks=[mc,rop])
loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'b',color='green', label='Training loss')

plt.plot(epochs, val_loss, 'b', color='red',label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.figure()

dc = history.history['dice_coef']

val_dc = history.history['val_dice_coef']

plt.plot(epochs, dc, 'b',color='green', label='Training Dice Coef.')

plt.plot(epochs, val_dc, 'b', color='red',label='Validation Dice Coef.')

plt.title('Training and validation Dice Coef.')

plt.legend()

plt.show()
def make_testdata(a):



    data = []

    c = 1



    for i in range(a.shape[0]-1):

        if a[i]+1 == a[i+1]:

            c += 1

            if i == a.shape[0]-2:

                data.append(str(a[i-c+2]))

                data.append(str(c))



        if a[i]+1 != a[i+1]:

            data.append(str(a[i-c+1]))

            data.append(str(c))

            c = 1



    data = " ".join(data)

    return data
#test Images

test_df=pd.read_csv(test_df_path)

model.load_weights('seg_model.h5')



enc_masks=[]

for ix in tqdm(range(0,test_df.shape[0],4)):

    img_ix=test_df.loc[ix,'ImageId_ClassId']

    img_ix=img_ix.split('_')[0]

    img=cv2.imread(os.path.join(test_img_path,img_ix))

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img=img.astype(np.float32) / 255.

    img=cv2.resize(img,(HEIGHT,WIDTH))

    #img=np.expand_dims(img,-1)

    img=np.expand_dims(img,0)

    pred_mask=model.predict(img)

    pred_mask=cv2.resize(pred_mask[0],(1600,256))

    for i in range(4):

        pred_fi = pred_mask[:,:,i].T.flatten()

        pred_fi = np.where(pred_fi > 0.5, 1, 0)

        pred_fi_id = np.where(pred_fi == 1)

        pred_fi_id = make_testdata(pred_fi_id[0])

        x = [img_ix + "_" + str(i+1), pred_fi_id]

        enc_masks.append(x)

    
columns = ['ImageId_ClassId', 'EncodedPixels']

d = pd.DataFrame(data=enc_masks, columns=columns, dtype='str')

d.to_csv("submission.csv",index=False)

print(d)