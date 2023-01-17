!pip install segmentation-models --quiet
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
train_img_path='../input/understanding_cloud_organization/train_images/'

test_img_path='../input/understanding_cloud_organization/test_images/'
BATCH_SIZE=100

LEARNING_RATE=0.00001

TARGET_HEIGHT,TARGET_WIDTH=192,192
train_df=pd.read_csv('../input/understanding_cloud_organization/train.csv')

test_df=pd.read_csv('../input/understanding_cloud_organization/sample_submission.csv')
train_df.head()
train_df['Image_Label_']=[train_img_path+ix for ix in train_df['Image_Label']]

#Fill Empty Encoding with 0

train_df['EncodedPixels'].fillna(0,inplace=True)
#Create list of list containing image index with their respective encoding

train_data=[]

for ix in tqdm(range(0,train_df.shape[0],4)):

    tmp=[]

    tmp.append(train_df.loc[ix,'Image_Label'].split('_')[0])

    for j in range(ix,ix+4):

        tmp.append(train_df.loc[j,'EncodedPixels'])

    train_data.append(tmp)
#Original Height,Width=1400,2100

def rleToMask(rleString,height,width,h=128,w=128,resize=False):

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
mask=rleToMask(train_data[0][1],1400,2100)
class_color=['Reds','Blues','Greens','Oranges']

fig=plt.figure(figsize=(30,30))

rows,cols=6,1

for i in range(1,rows*cols+1):

    img=cv2.imread(os.path.join(train_img_path,train[i-1][0]))

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    fig.add_subplot(rows,cols,i)

    plt.imshow(img)

    for j in range(4):

        msk_encode=train[i-1][j+1]

        if msk_encode==0:

            continue

        else:

            mask=rleToMask(msk_encode,1400,2100)

            plt.imshow(mask,cmap=class_color[j],alpha=0.4)            

plt.show()
class customGenerator(Sequence):

    def __init__(self,data_list,batch_size,target_height,target_width,is_train=True,img_path=train_img_path):

        self.indexes,self.mask_ids=sep_indexes(data_list)

        self.batch_size=batch_size

        self.height=target_height

        self.width=target_width

        self.is_train=is_train

        self.img_path=train_img_path

    

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

    

    def aug_fx(self,image,mask):

        aug = Compose([

        OneOf([RandomSizedCrop(min_max_height=(50, 101), height=self.height, width=self.width, p=0.5),

              PadIfNeeded(min_height=self.height, min_width=self.width, p=0.5)], p=1),    

        VerticalFlip(p=0.5),              

        RandomRotate90(p=0.5),

        OneOf([

            ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),

            GridDistortion(p=0.5),

            OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)                  

            ], p=0.8),



        RandomBrightnessContrast(p=0.8),    

        RandomGamma(p=0.8)])

        augmented = aug(image=image, mask=mask)

        return augmented['image'],augmented['mask']

    

    def load_images(self,img_ids):

        tmp=np.zeros((len(img_ids),self.height,self.width,3))

        for ix,id_ in enumerate(img_ids):

            img=cv2.imread(os.path.join(self.img_path,id_))

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

                    mask=rleToMask(enc_,1400,2100,self.height,self.width,resize=True)

                    tmp[ix,:,:,j]=mask

        return tmp

    

    def train_generator(self,batch_x,batch_y):

        image_batch=self.load_images(batch_x)

        mask_batch=self.load_masks(batch_y)

        

        #Augmentation

        for ix in range(len(image_batch)):

            image_batch[ix],mask_batch[ix]=self.aug_fx(image_batch[ix],mask_batch[ix])

            

        return image_batch,mask_batch

    

    def val_generator(self,batch_x,batch_y):

        image_batch=self.load_images(batch_x)

        mask_batch=self.load_masks(batch_y)

        return image_batch,mask_batch
train_gen=customGenerator(train,BATCH_SIZE,TARGET_HEIGHT,TARGET_WIDTH)

val_gen=customGenerator(val,BATCH_SIZE,TARGET_HEIGHT,TARGET_WIDTH,is_train=False)
import segmentation_models as sm

preprocess_input = sm.backbones.get_preprocessing('resnet50')

model=sm.Unet('resnet50',input_shape=(TARGET_HEIGHT,TARGET_WIDTH,3),classes=4,activation='sigmoid')

model.compile(loss=sm.losses.dice_loss,optimizer=Adam(0.00002),metrics=[sm.metrics.dice_score])
train_steps=int(np.ceil(len(train)/BATCH_SIZE))

val_steps=int(np.ceil(len(val)/BATCH_SIZE))
mc=ModelCheckpoint('cloud_seg_2.h5',monitor='val_loss',mode='min',save_best_only=True,period=1,verbose=1)

rop=ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,min_lr=0.0000001)
history=model.fit_generator(train_gen,epochs=5,steps_per_epoch=train_steps,

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

model.load_weights('cloud_seg_2.h5')



enc_masks=[]

for ix in tqdm(range(0,test_df.shape[0],4)):

    img_ix=test_df.loc[ix,'Image_Label']

    img_ix=img_ix.split('_')[0]

    img=cv2.imread(os.path.join(test_img_path,img_ix))

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img=img.astype(np.float32) / 255.

    img=cv2.resize(img,(TARGET_HEIGHT,TARGET_WIDTH))

    #img=np.expand_dims(img,-1)

    img=np.expand_dims(img,0)

    pred_mask=model.predict(img)

    pred_mask=cv2.resize(pred_mask[0],(2100,1400))

    for i in range(4):

        pred_fi = pred_mask[:,:,i].T.flatten()

        pred_fi = np.where(pred_fi > 0.5, 1, 0)

        pred_fi_id = np.where(pred_fi == 1)

        pred_fi_id = make_testdata(pred_fi_id[0])

        x = [img_ix + "_" + str(i+1), pred_fi_id]

        enc_masks.append(x)
columns = ['Image_Label', 'EncodedPixels']

d = pd.DataFrame(data=enc_masks, columns=columns, dtype='str')

d['Image_Label']=test_df['Image_Label']

d.to_csv("submission.csv",index=False)

df = pd.read_csv("submission.csv")

print(df)