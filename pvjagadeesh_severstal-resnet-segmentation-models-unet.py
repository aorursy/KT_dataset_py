import os

import json

import cv2

import keras

from keras import backend as K

from PIL import Image

import matplotlib.pyplot as plt,time

import numpy as np

import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split
path = '../input/severstal-steel-defect-detection/'
train = pd.read_csv(path + 'train.csv')



train['ImageId'] = train['ImageId_ClassId'].map(lambda x: x.split('.')[0]+'.jpg')

train2 = pd.DataFrame({'ImageId':train['ImageId'][::4]})

train2['e1'] = train['EncodedPixels'][::4].values

train2['e2'] = train['EncodedPixels'][1::4].values

train2['e3'] = train['EncodedPixels'][2::4].values

train2['e4'] = train['EncodedPixels'][3::4].values

train2.reset_index(inplace=True,drop=True)

train2.fillna('',inplace=True); 

train2['count'] = np.sum(train2.iloc[:,1:]!='',axis=1).values

train2.head()
class DataGenerator(keras.utils.Sequence):

    def __init__(self, df, batch_size = 16, subset="train", shuffle=False, 

                 preprocess=None, info={}):

        super().__init__()

        self.df = df

        self.shuffle = shuffle

        self.subset = subset

        self.batch_size = batch_size

        self.preprocess = preprocess

        self.info = info

        

        if self.subset == "train":

            self.data_path = path + 'train_images/'

        elif self.subset == "test":

            self.data_path = path + 'test_images/'

        self.on_epoch_end()



    def __len__(self):

        return int(np.floor(len(self.df) / self.batch_size))

    

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.df))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)

    

    def __getitem__(self, index): 

        X = np.empty((self.batch_size,128,800,3),dtype=np.float32)

        y = np.empty((self.batch_size,128,800,4),dtype=np.int8)

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        for i,f in enumerate(self.df['ImageId'].iloc[indexes]):

            self.info[index*self.batch_size+i]=f

            X[i,] = Image.open(self.data_path + f).resize((800,128))

            if self.subset == 'train': 

                for j in range(4):

                    y[i,:,:,j] = rle2maskResize(self.df['e'+str(j+1)].iloc[indexes[i]])

        if self.preprocess!=None: X = self.preprocess(X)

        if self.subset == 'train': return X, y

        else: return X
def rle2maskResize(rle):

    # CONVERT RLE TO MASK 

    if (pd.isnull(rle))|(rle==''): 

        return np.zeros((128,800) ,dtype=np.uint8)

    

    height= 256

    width = 1600

    mask= np.zeros( width*height ,dtype=np.uint8)



    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]-1

    lengths = array[1::2]    

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

    

    return mask.reshape( (height,width), order='F' )[::2,::2]

  

def rle2mask(rle, imgshape):

    width = imgshape[0]

    height= imgshape[1]

    

    mask= np.zeros( width*height ).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

        current_position += lengths[index]

        

    return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )



def mask2contour(mask, width=3):

    # CONVERT MASK TO ITS CONTOUR

    w = mask.shape[1]

    h = mask.shape[0]

    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)

    mask2 = np.logical_xor(mask,mask2)

    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)

    mask3 = np.logical_xor(mask,mask3)

    return np.logical_or(mask2,mask3) 



def mask2pad(mask, pad=2):

    # ENLARGE MASK TO INCLUDE MORE SPACE AROUND DEFECT

    w = mask.shape[1]

    h = mask.shape[0]

    

    # MASK UP

    for k in range(1,pad,2):

        temp = np.concatenate([mask[k:,:],np.zeros((k,w))],axis=0)

        mask = np.logical_or(mask,temp)

    # MASK DOWN

    for k in range(1,pad,2):

        temp = np.concatenate([np.zeros((k,w)),mask[:-k,:]],axis=0)

        mask = np.logical_or(mask,temp)

    # MASK LEFT

    for k in range(1,pad,2):

        temp = np.concatenate([mask[:,k:],np.zeros((h,k))],axis=1)

        mask = np.logical_or(mask,temp)

    # MASK RIGHT

    for k in range(1,pad,2):

        temp = np.concatenate([np.zeros((h,k)),mask[:,:-k]],axis=1)

        mask = np.logical_or(mask,temp)

    

    return mask
def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
! pip install segmentation-models
import segmentation_models as sm

from segmentation_models.backbones import get_preprocessing



BACKBONE = 'resnet34'

preprocess_input = get_preprocessing(BACKBONE)



model = sm.Unet(BACKBONE, input_shape=(128, 800, 3),classes=4, activation='sigmoid')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])



# TRAIN AND VALIDATE MODEL

idx = int(0.8*len(train2)); print()

train_batches = DataGenerator(train2.iloc[:idx],shuffle=True,preprocess=preprocess_input)

valid_batches = DataGenerator(train2.iloc[idx:],preprocess=preprocess_input)

#history = model.fit_generator(train_batches, validation_data = valid_batches, epochs = 30, verbose=1)

#history = model.fit_generator(train_batches, validation_data = valid_batches, epochs = 30, verbose=1)
#plt.figure(figsize=(8, 8))

#plt.title("Learning curve")

#plt.plot(history.history["loss"], label="loss")

#plt.plot(history.history["val_loss"], label="val_loss")

#plt.plot( np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r", label="best model")

#plt.xlabel("Epochs")

#plt.ylabel("log_loss")

##plt.legend();
#plt.figure(figsize=(15,5))

#plt.plot(range(history.epoch[-1]+1),history.history['val_dice_coef'],label='val_dice_coef')

#plt.plot(range(history.epoch[-1]+1),history.history['dice_coef'],label='trn_dice_coef')

#plt.title('Training Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Dice_coef');plt.legend(); 

#plt.show()
#model.save('RasNet+Unet.h5')
!ls