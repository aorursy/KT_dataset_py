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
pd.set_option("display.max_rows", 101)

import cv2

import json

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams["font.size"] = 15

import seaborn as sns

from collections import Counter

from PIL import Image

import math

import seaborn as sns

from collections import defaultdict

from pathlib import Path

import cv2

from tqdm import tqdm
train_df = pd.read_csv("../input/train.csv")

sample_df = pd.read_csv("../input/sample_submission.csv")
train_df.head()
# check number of each class



class_dict = defaultdict(int)



kind_class_dict = defaultdict(int)



no_defects_num = 0

defects_num = 0



for col in range(0, len(train_df), 4):

    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]

    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):

        raise ValueError

        

    labels = train_df.iloc[col:col+4, 1]

    if labels.isna().all():

        no_defects_num += 1

    else:

        defects_num += 1

    

    kind_class_dict[sum(labels.isna().values == False)] += 1

        

    for idx, label in enumerate(labels.isna().values.tolist()):

        if label == False:

            class_dict[idx+1] += 1
print("the number of images with no defects: {}".format(no_defects_num))

print("the number of images with defects: {}".format(defects_num))
fig, ax = plt.subplots()

sns.barplot(x=list(class_dict.keys()), y=list(class_dict.values()), ax=ax)

ax.set_title("the number of images for each class")

ax.set_xlabel("class")

class_dict
fig, ax = plt.subplots()

sns.barplot(x=list(kind_class_dict.keys()), y=list(kind_class_dict.values()), ax=ax)

ax.set_title("Number of classes included in each image");

ax.set_xlabel("number of classes in the image")

kind_class_dict
train_size_dict = defaultdict(int)

train_path = Path("../input/train_images/")



for img_name in train_path.iterdir():

    img = Image.open(img_name)

    train_size_dict[img.size] += 1
train_size_dict
test_size_dict = defaultdict(int)

test_path = Path("../input/test_images/")



for img_name in test_path.iterdir():

    img = Image.open(img_name)

    test_size_dict[img.size] += 1
test_size_dict
palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]
def name_and_mask(start_idx):

    col = start_idx

    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]

    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):

        raise ValueError



    labels = train_df.iloc[col:col+4, 1]

    mask = np.zeros((256, 1600, 4), dtype=np.uint8)



    for idx, label in enumerate(labels.values):

        if label is not np.nan:

            mask_label = np.zeros(1600*256, dtype=np.uint8)

            label = label.split(" ")

            positions = map(int, label[0::2])

            length = map(int, label[1::2])

            for pos, le in zip(positions, length):

                mask_label[pos:(pos+le)] = 1

            mask[:, :, idx] = mask_label.reshape(256, 1600, order='F')

    return img_names[0], mask
def show_mask_image(col):

    name, mask = name_and_mask(col)

    img = cv2.imread(str(train_path / name))

    fig, ax = plt.subplots(figsize=(15, 15))



    for ch in range(4):

        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for i in range(0, len(contours)):

            cv2.polylines(img, contours[i], True, palet[ch], 2)

    ax.set_title(name)

    ax.imshow(img)

    plt.show()
fig, ax = plt.subplots(1, 4, figsize=(15, 5))

for i in range(4):

    ax[i].axis('off')

    ax[i].imshow(np.ones((50, 50, 3), dtype=np.uint8) * palet[i])

    ax[i].set_title("class color: {}".format(i+1))

fig.suptitle("each class colors")



plt.show()
idx_no_defect = []

idx_class_1 = []

idx_class_2 = []

idx_class_3 = []

idx_class_4 = []

idx_class_multi = []

idx_class_triple = []



for col in range(0, len(train_df), 4):

    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]

    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):

        raise ValueError

        

    labels = train_df.iloc[col:col+4, 1]

    if labels.isna().all():

        idx_no_defect.append(col)

    elif (labels.isna() == [False, True, True, True]).all():

        idx_class_1.append(col)

    elif (labels.isna() == [True, False, True, True]).all():

        idx_class_2.append(col)

    elif (labels.isna() == [True, True, False, True]).all():

        idx_class_3.append(col)

    elif (labels.isna() == [True, True, True, False]).all():

        idx_class_4.append(col)

    elif labels.isna().sum() == 1:

        idx_class_triple.append(col)

    else:

        idx_class_multi.append(col)
for idx in idx_no_defect[:5]:

    show_mask_image(idx)
for idx in idx_class_1[:5]:

    show_mask_image(idx)
for idx in idx_class_2[:5]:

    show_mask_image(idx)
for idx in idx_class_3[:5]:

    show_mask_image(idx)
for idx in idx_class_4[:5]:

    show_mask_image(idx)
for idx in idx_class_multi[:5]:

    show_mask_image(idx)
for idx in idx_class_triple:

    show_mask_image(idx)
for col in tqdm(range(0, len(train_df), 4)):

    name, mask = name_and_mask(col)

    if (mask.sum(axis=2) >= 2).any():

        show_mask_image(idx)
train_df = train_df[ train_df['EncodedPixels'].notnull() ]

print( train_df.shape )

train_df.head()
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

        

    return np.flipud( np.rot90( mask.reshape(height,width), k=1 ) )



fig=plt.figure(figsize=(20,100))

columns = 2

rows = 50

for i in range(1, 100+1):

    fig.add_subplot(rows, columns, i)

    

    fn = train_df['ImageId_ClassId'].iloc[i].split('_')[0]

    img = cv2.imread( '../input/train_images/'+fn )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = rle2mask(train_df['EncodedPixels'].iloc[i], img.shape  )

    img[mask==1,0] = 255

    

    plt.imshow(img)

plt.show()
import gc

import time

from PIL import Image 

import keras

import tensorflow as tf

from keras import backend as K

from keras.losses import binary_crossentropy

import warnings

warnings.filterwarnings("ignore")



path = '../input/'

train = pd.read_csv(path + 'train.csv')



# RESTRUCTURE TRAIN DATAFRAME

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
# https://www.kaggle.com/ateplyuk/pytorch-starter-u-net-resnet

# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly



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
# https://www.kaggle.com/titericz/building-and-visualizing-masks

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
plt.figure(figsize=(13.5,2.5))

bar = plt.bar( [1,2,3,4],100*np.mean( train2.iloc[:,1:5]!='',axis=0) )

plt.title('Percent Training Images with Defect', fontsize=16)

plt.ylabel('Percent of Images'); plt.xlabel('Defect Type')

plt.xticks([1,2,3,4])

for rect in bar:

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f %%' % height,

             ha='center', va='bottom',fontsize=16)

plt.ylim((0,50)); plt.show()
# DEFECTIVE IMAGE SAMPLES

filenames = {}

defects = list(train2[train2['e1']!=''].sample(3).index)

defects += list(train2[train2['e2']!=''].sample(3).index)

defects += list(train2[train2['e3']!=''].sample(7).index)

defects += list(train2[train2['e4']!=''].sample(3).index)



# DATA GENERATOR

train_batches = DataGenerator(train2[train2.index.isin(defects)],shuffle=True,info=filenames)

print('Images and masks from our Data Generator')

print('KEY: yellow=defect1, green=defect2, blue=defect3, magenta=defect4')



# DISPLAY IMAGES WITH DEFECTS

for i,batch in enumerate(train_batches):

    plt.figure(figsize=(14,50)) #20,18

    for k in range(16):

        plt.subplot(16,1,k+1)

        img = batch[0][k,]

        img = Image.fromarray(img.astype('uint8'))

        img = np.array(img)

        extra = '  has defect'

        for j in range(4):

            msk = batch[1][k,:,:,j]

            msk = mask2pad(msk,pad=3)

            msk = mask2contour(msk,width=2)

            if np.sum(msk)!=0: extra += ' '+str(j+1)

            if j==0: # yellow

                img[msk==1,0] = 235 

                img[msk==1,1] = 235

            elif j==1: img[msk==1,1] = 210 # green

            elif j==2: img[msk==1,2] = 255 # blue

            elif j==3: # magenta

                img[msk==1,0] = 255

                img[msk==1,2] = 255

        plt.title(filenames[16*i+k]+extra)

        plt.axis('off') 

        plt.imshow(img)

    plt.subplots_adjust(wspace=0.05)

    plt.show()
! pip install segmentation-models
# COMPETITION METRIC

def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



# LOSS FUNCTION



def dice_loss(y_true, y_pred):

    loss = 1 - dice_coef(y_true, y_pred)

    return loss



def bce_dice_loss(y_true, y_pred):

    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    return loss



# Focal Tversky loss, brought to you by:  https://github.com/nabsabraham/focal-tversky-unet

def tversky(y_true, y_pred, smooth=1e-6):

    y_true_pos = tf.keras.layers.Flatten()(y_true)

    y_pred_pos = tf.keras.layers.Flatten()(y_pred)

    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)

    false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))

    false_pos = tf.reduce_sum((1-y_true_pos)*y_pred_pos)

    alpha = 0.7

    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)



def tversky_loss(y_true, y_pred):

    return 1 - tversky(y_true,y_pred)



def focal_tversky_loss(y_true,y_pred):

    pt_1 = tversky(y_true, y_pred)

    gamma = 0.75

    return tf.keras.backend.pow((1 - pt_1), gamma)
from segmentation_models import Unet

from segmentation_models.backbones import get_preprocessing



# LOAD UNET WITH PRETRAINING FROM IMAGENET

preprocess = get_preprocessing('resnet34') # for resnet, img = (img-110.0)/1.0

model = Unet('resnet34', input_shape=(128, 800, 3), classes=4, activation='sigmoid')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])

#adam = tf.keras.optimizers.Adam(lr = 0.05, epsilon = 0.1)

#model.complie(optimizer = adam, loss = focal_tversky_loss, metrics = [tversky, dice_coef])



# TRAIN AND VALIDATE MODEL

idx = int(0.8*len(train2)); print()

train_batches = DataGenerator(train2.iloc[:idx],shuffle = True, preprocess = preprocess)

valid_batches = DataGenerator(train2.iloc[idx:],preprocess=preprocess)

history = model.fit_generator(train_batches, validation_data = valid_batches, epochs = 30, verbose = 2)
# PLOT TRAINING

plt.figure(figsize=(15,5))

plt.plot(range(history.epoch[-1]+1),history.history['val_dice_coef'],label='val_dice_coef')

plt.plot(range(history.epoch[-1]+1),history.history['dice_coef'],label='trn_dice_coef')

plt.title('Training Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Dice_coef');plt.legend(); 

plt.show()
# PREDICT FROM VALIDATION SET (ONLY IMAGES WITH DEFECTS)

val_set = train2.iloc[idx:];

defects = list(val_set[val_set['e1']!=''].sample(6).index)

defects += list(val_set[val_set['e2']!=''].sample(6).index)

defects += list(val_set[val_set['e3']!=''].sample(14).index)

defects += list(val_set[val_set['e4']!=''].sample(6).index)



valid_batches = DataGenerator(val_set[val_set.index.isin(defects)],preprocess=preprocess)

preds = model.predict_generator(valid_batches,verbose=1)
# PLOT PREDICTIONS

valid_batches = DataGenerator(val_set[val_set.index.isin(defects)])

print('Plotting predictions...')

print('KEY: yellow=defect1, green=defect2, blue=defect3, magenta=defect4')



for i,batch in enumerate(valid_batches):

    plt.figure(figsize=(20,36))

    for k in range(16):

        plt.subplot(16,2,2*k+1)

        img = batch[0][k,]

        img = Image.fromarray(img.astype('uint8'))

        img = np.array(img)

        dft = 0

        extra = '  has defect '

        for j in range(4):

            msk = batch[1][k,:,:,j]

            if np.sum(msk)!=0: 

                dft=j+1

                extra += ' '+str(j+1)

            msk = mask2pad(msk,pad=2)

            msk = mask2contour(msk,width=3)

            if j==0: # yellow

                img[msk==1,0] = 235 

                img[msk==1,1] = 235

            elif j==1: img[msk==1,1] = 210 # green

            elif j==2: img[msk==1,2] = 255 # blue

            elif j==3: # magenta

                img[msk==1,0] = 255

                img[msk==1,2] = 255

        if extra=='  has defect ': extra =''

        plt.title('Train '+train2.iloc[16*i+k,0]+extra)

        plt.axis('off') 

        plt.imshow(img)

        plt.subplot(16,2,2*k+2) 

        if dft!=0:

            msk = preds[16*i+k,:,:,dft-1]

            plt.imshow(msk)

        else:

            plt.imshow(np.zeros((128,800)))

        plt.axis('off')

        mx = np.round(np.max(msk),3)

        plt.title('Predict Defect '+str(dft)+'  (max pixel = '+str(mx)+')')

    plt.subplots_adjust(wspace=0.05)

    plt.show()
# PREDICT FROM VALIDATION SET (ONLY IMAGES WITH DEFECTS 1, 2, 4)

val_set = train2.iloc[idx:]

val_set2 = val_set[(val_set['count']!=0)&(val_set['e3']=='')].sample(16)



valid_batches = DataGenerator(val_set2,preprocess=preprocess)

preds = model.predict_generator(valid_batches,verbose=1)
# PLOT PREDICTIONS

valid_batches = DataGenerator(val_set2)

print('Plotting predictions...')

print('KEY: yellow=defect1, green=defect2, blue=defect3, magenta=defect4')

for i,batch in enumerate(valid_batches):

    plt.figure(figsize=(20,36))

    for k in range(16):

        plt.subplot(16,2,2*k+1)

        img = batch[0][k,]

        img = Image.fromarray(img.astype('uint8'))

        img = np.array(img)

        dft = 0

        three = False

        for j in range(4):

            msk = batch[1][k,:,:,j]

            if (j==2)&(np.sum(msk)!=0): 

                three=np.sum(msk)

            msk = mask2pad(msk,pad=2)

            msk = mask2contour(msk,width=3)

            if j==0: # yellow

                img[msk==1,0] = 235 

                img[msk==1,1] = 235

            elif j==1: img[msk==1,1] = 210 # green

            elif j==2: img[msk==1,2] = 255 # blue

            elif j==3: # magenta

                img[msk==1,0] = 255

                img[msk==1,2] = 255

        extra = ''; extra2 = ''

        if not three: 

            extra = 'NO DEFECT 3'

            extra2 = 'ERROR '

        plt.title('Train '+train2.iloc[16*i+k,0]+'  '+extra)

        plt.axis('off') 

        plt.imshow(img)

        plt.subplot(16,2,2*k+2) 

        dft=3

        if dft!=0:

            msk = preds[16*i+k,:,:,dft-1]

            plt.imshow(msk)

        else:

            plt.imshow(np.zeros((128,800)))

        plt.axis('off')

        mx = np.round(np.max(msk),3)

        plt.title(extra2+'Predict Defect '+str(dft)+'  (max pixel = '+str(mx)+')')

    plt.subplots_adjust(wspace=0.05)

    plt.show()
# PREDICT FROM VALIDATION SET (USE ALL)

valid_batches = DataGenerator(train2.iloc[idx:],preprocess=preprocess)

preds = model.predict_generator(valid_batches,verbose=1)
# PLOT RESULTS

import seaborn as sns

pix_min = 250

for THRESHOLD in [0.1, 0.25, 0.50, 0.75, 0.9]:

    print('######################################')

    print('## Threshold =',THRESHOLD,'displayed below ##')

    print('######################################')

    correct=[[],[],[],[]]; incorrect=[[],[],[],[]]

    for i,f in enumerate(train2.iloc[idx:idx+len(preds)]['ImageId']):

        preds2 = preds[i].copy()

        preds2[preds2>=THRESHOLD]=1

        preds2[preds2<THRESHOLD]=0

        sums = np.sum(preds2,axis=(0,1))

        for j in range(4):

            if 4*sums[j]<pix_min: continue

            if train2.iloc[i,j+1]=='': incorrect[j].append(4*sums[j])

            else: correct[j].append(4*sums[j])

    plt.figure(figsize=(20,8))

    for j in range(4):

        limit = [10000,10000,100000,100000][j]

        plt.subplot(2,2,j+1)

        sns.distplot([x for x in correct[j] if x<limit], label = 'correct')

        sns.distplot([x for x in incorrect[j] if x<limit], label = 'incorrect')

        plt.title('Defect '+str(j+1)+' mask sizes with threshold = '+str(THRESHOLD)); plt.legend()

    plt.show()

    for j in range(4):

        c1 = np.array(correct[j])

        c2 = np.array(incorrect[j])

        print('With threshold =',THRESHOLD,', defect',j+1,'has',len(c1[c1!=0]),'correct and',len(c2[c2!=0]),'incorrect masks')

    print()
# SAVE MODEL

model.save('UNET.h5')



# LOAD MODEL

from keras.models import load_model

model = load_model('UNET.h5',custom_objects={'dice_coef': dice_coef})



# PREDICT 1 BATCH TEST DATASET

test = pd.read_csv(path + 'sample_submission.csv')

test['ImageId'] = test['ImageId_ClassId'].map(lambda x: x.split('_')[0])

test_batches = DataGenerator(test.iloc[::4],subset='test',batch_size=256,preprocess=preprocess)

test_preds = model.predict_generator(test_batches,steps=1,verbose=1)