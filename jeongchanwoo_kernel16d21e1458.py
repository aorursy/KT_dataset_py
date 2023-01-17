# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import cv2

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
model_weight = '/kaggle/input/unet-exception/unet_exception_15-0.0138'

path = '/kaggle/input/severstal-steel-defect-detection/'
mask_threshold = 512

mask_bound_1 = 0.10

mask_bound_2 = 0.05

mask_bound_3 = 0.5

mask_bound_4 = 0.5

import gc

import matplotlib.pyplot as plt, time

from PIL import Image

import keras
# COMPETITION METRIC

def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

class DataGenerator(keras.utils.Sequence):

    def __init__(self, df, batch_size = 16 ,subset ='train', shuffle = False, preprocess = None, info={}):

        super().__init__()

        self.df = df

        self.shuffle = shuffle

        self.subset = subset

        self.batch_size = batch_size

        self.preprocess = preprocess

        self.info = info

        

        if self.subset =='train':

            self.data_path = path +'train_images/'

#         elif self.subset =='valid':

#             self.data_path = path +'train_images/'

        elif self.subset =='test':

            self.data_path = path + 'test_images/'

        self.on_epoch_end()

        

    def __len__(self):

        return int(np.floor(len(self.df) / self.batch_size))

    

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.df))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)

    def __getitem__(self,index):

        x = np.empty((self.batch_size, 128, 800, 3), dtype=np.float32)

        y = np.empty((self.batch_size, 128, 800, 4), dtype=np.int8)

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        for i,f in enumerate(self.df['ImageId'].iloc[indexes]):

            self.info[index*self.batch_size + i] =f 

            x[i,]=Image.open(self.data_path + f).resize((800,128))

            if self.subset =='train':

                for j in range(4):

                    y[i,:,:,j] = rle2maskResize(self.df['e'+str(j+1)].iloc[indexes[i]])

        if self.preprocess !=None : x= self.preprocess(x)

        if self.subset == 'train' : return x,y

        else: return x
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
from keras.models import Model, load_model

from keras.layers import Input,Dropout,BatchNormalization,Activation,Add

from keras.layers.core import Lambda

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras import backend as K



import tensorflow as tf

# config = tf.ConfigProto()

# config.gpu_options.per_process_gpu_memory_fraction = 0.4

# session = tf.Session(config= config)



from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



from keras.models import load_model
def mask2rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)

def build_rles(masks):

    width, height, depth = masks.shape

    

    rles = [mask2rle(masks[:, :, i])

            for i in range(depth)]

    

    return rles
def masks_reduce(masks):

    

    for idx in range(masks.shape[-1]):

        label_num, labeled_mask = cv2.connectedComponents(masks[:,:,idx].astype(np.uint8))

        reduced_mask = np.zeros(masks.shape[:2],np.float32)

        

        for label in range(1, label_num):

            single_label_mask = (labeled_mask == label)

            if single_label_mask.sum() > mask_threshold:

                reduced_mask[single_label_mask] = 1

        

        masks[:,:,idx] = reduced_mask

        

    return masks



def masks_reduce2(masks):

    for idx in range(masks.shape[-1]):

        if np.sum(masks[:,:,idx]) < mask_threshold:

            masks[:,:,idx] = np.zeros(masks.shape[:2], dtype = np.uint8)

    return masks
model = load_model(model_weight, custom_objects={'dice_coef' : dice_coef})
# train = pd.read_csv(os.path.join(path, 'train.csv'))

# train['ImageId'] = train['ImageId_ClassId'].map(lambda x : x.split('.')[0] + '.jpg')

# train2 = pd.DataFrame({'ImageId' : train['ImageId'][::4]})

# train2['e1'] = train['EncodedPixels'][::4].values

# train2['e2'] = train['EncodedPixels'][1::4].values

# train2['e3'] = train['EncodedPixels'][2::4].values

# train2['e4'] = train['EncodedPixels'][3::4].values

# train2.reset_index(inplace=True, drop =True)

# train2.fillna('',inplace=True)

# train2['count'] = np.sum(train2.iloc[:,1:]!='', axis = 1).values
# idx = int(0.8*len(train2))

# train_batches = DataGenerator(train2.iloc[:idx],batch_size= 16, shuffle=True)

# valid_batches = DataGenerator(train2.iloc[idx:],batch_size= 16)
# valid_df_resize = []

# for i in range(0,train2.iloc[idx:].shape[0], 300):

#     batch_idx = list(range(i, min(train2.iloc[idx:].shape[0] , i+300)))

#     valid_generator = DataGenerator(

#         train2.iloc[idx:].iloc[batch_idx], subset = 'valid', batch_size = 1)

#     valid_preds =model.predict_generator(valid_generator, verbose = 1)

    

#     for j, b in tqdm(enumerate(batch_idx)):

#         filename = train2.iloc[idx:]['ImageId'].iloc[b]

#         image_df = train[train['ImageId'] == filename].copy()

        

#         pred_masks = np.squeeze(np.round(valid_preds[j,])).astype(np.uint8)

# #         pred_masks = test_preds[j, ].round().astype(int)

#         pred_masks_re = cv2.resize(pred_masks, (1600,256))



#         pred_rles = build_rles(pred_masks)

#         image_df['EncodedPixels'] = pred_rles

#         valid_df_resize.append(image_df)

#     gc.collect()
# valid_resize = pd.concat(valid_df_resize)
# valid_resize.reset_index(inplace=True, drop = True)
# val_set = train[train['ImageId'].isin(valid_batches.df['ImageId'].values)].copy()

# val_set.reset_index(inplace = True, drop = True)
# valid_resize.shape, val_set.shape
# valid_resize.head()
# val_set.head()
# val_set2 = pd.DataFrame({'ImageId' : val_set['ImageId'][::4]})

# val_set2['e1'] = val_set['EncodedPixels'][::4].values

# val_set2['e2'] = val_set['EncodedPixels'][1::4].values

# val_set2['e3'] = val_set['EncodedPixels'][2::4].values

# val_set2['e4'] = val_set['EncodedPixels'][3::4].values

# val_set2.reset_index(inplace=True, drop =True)

# val_set2.fillna('',inplace=True)

# val_set2['count'] = np.sum(val_set2.iloc[:,1:]!='', axis = 1).values

# val_set2.head()
# filenames = {}

# defects = list(val_set2[val_set2['e1']!=''].sample(3).index)

# defects += list(val_set2[val_set2['e2']!=''].sample(3).index)

# defects += list(val_set2[val_set2['e3']!=''].sample(7).index)

# defects += list(val_set2[val_set2['e4']!=''].sample(3).index)
# valid_batches = DataGenerator(val_set2[val_set2.index.isin(defects)],batch_size= 16, shuffle=False,info=filenames )
# for i, batch in enumerate(valid_batches):

#     plt.figure(figsize=(14,50))

#     for k in range(16):

#         plt.subplot(16,1, k+1)

#         img = batch[0][k,]

#         img = Image.fromarray(img.astype('uint8'))

#         img = np.array(img)

# #         print(img.shape)

#         extra = ' has defect'

#         for j in range(4):

#             msk = batch[1][k, : , : , j]

#             msk = mask2pad(msk, pad =3)

#             msk = mask2contour(msk, width =2)

#             if np.sum(msk)!=0 :

#                 extra +=' ' + str(j+1)

#             if j==0:

#                 img[msk==1,0]==235

#                 img[msk==1,1]=235

#             elif j==1:

#                 img[msk==1,1]=210

#             elif j==2:

#                 img[msk==1,2]=255

#             elif j==3:

#                 img[msk==1,0]=255

#                 img[msk==1,2]=255

#         plt.title(filenames[16*i+k] + extra)

#         plt.axis('off')

#         plt.imshow(img)

#     plt.subplots_adjust(wspace = 0.05)

#     plt.show()
# class DataGenerator_Resize(keras.utils.Sequence):

#     def __init__(self, df, batch_size = 16 ,subset ='train', shuffle = False, preprocess = None, info={}):

#         super().__init__()

#         self.df = df

#         self.shuffle = shuffle

#         self.subset = subset

#         self.batch_size = batch_size

#         self.preprocess = preprocess

#         self.info = info

        

#         if self.subset =='train':

#             self.data_path = path +'train_images/'

#         elif self.subset =='valid':

#             self.data_path = path +'train_images/'

#         elif self.subset =='test':

#             self.data_path = path + 'test_images/'

#         self.on_epoch_end()

        

#     def __len__(self):

#         return int(np.floor(len(self.df) / self.batch_size))

    

#     def on_epoch_end(self):

#         self.indexes = np.arange(len(self.df))

#         if self.shuffle == True:

#             np.random.shuffle(self.indexes)

#     def __getitem__(self,index):

#         x = np.empty((self.batch_size, 256, 1600, 3), dtype=np.float32)

#         y = np.empty((self.batch_size, 256, 1600, 4), dtype=np.int8)

#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         for i,f in enumerate(self.df['ImageId'].iloc[indexes]):

#             self.info[index*self.batch_size + i] =f 

#             x[i,]=Image.open(self.data_path + f).resize((1600,256))

#             if self.subset =='train':

#                 for j in range(4):

#                     y[i,:,:,j] = rle2maskResize(self.df['e'+str(j+1)].iloc[indexes[i]])

#         if self.preprocess !=None : x= self.preprocess(x)

#         if self.subset == 'train' : return x,y

#         else: return x
# def rle2maskResize(rle):

#     # CONVERT RLE TO MASK 

#     if (pd.isnull(rle))|(rle==''): 

#         return np.zeros((256,1600) ,dtype=np.uint8)

    

#     height= 256

#     width = 1600

#     mask= np.zeros( width*height ,dtype=np.uint8)



#     array = np.asarray([int(x) for x in rle.split()])

#     starts = array[0::2]-1

#     lengths = array[1::2]    

#     for index, start in enumerate(starts):

#         mask[int(start):int(start+lengths[index])] = 1

    

#     return mask.reshape( (height,width), order='F' )[::1,::1]



# def mask2contour(mask, width=3):

#     # CONVERT MASK TO ITS CONTOUR

#     w = mask.shape[1]

#     h = mask.shape[0]

#     mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)

#     mask2 = np.logical_xor(mask,mask2)

#     mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)

#     mask3 = np.logical_xor(mask,mask3)

#     return np.logical_or(mask2,mask3) 



# def mask2pad(mask, pad=2):

#     # ENLARGE MASK TO INCLUDE MORE SPACE AROUND DEFECT

#     w = mask.shape[1]

#     h = mask.shape[0]

    

#     # MASK UP

#     for k in range(1,pad,2):

#         temp = np.concatenate([mask[k:,:],np.zeros((k,w))],axis=0)

#         mask = np.logical_or(mask,temp)

#     # MASK DOWN

#     for k in range(1,pad,2):

#         temp = np.concatenate([np.zeros((k,w)),mask[:-k,:]],axis=0)

#         mask = np.logical_or(mask,temp)

#     # MASK LEFT

#     for k in range(1,pad,2):

#         temp = np.concatenate([mask[:,k:],np.zeros((h,k))],axis=1)

#         mask = np.logical_or(mask,temp)

#     # MASK RIGHT

#     for k in range(1,pad,2):

#         temp = np.concatenate([np.zeros((h,k)),mask[:,:-k]],axis=1)

#         mask = np.logical_or(mask,temp)

    

#     return mask 
# val_resize2 = pd.DataFrame({'ImageId' : valid_resize['ImageId'][::4]})

# val_resize2['e1'] = valid_resize['EncodedPixels'][::4].values

# val_resize2['e2'] = valid_resize['EncodedPixels'][1::4].values

# val_resize2['e3'] = valid_resize['EncodedPixels'][2::4].values

# val_resize2['e4'] = valid_resize['EncodedPixels'][3::4].values

# val_resize2.reset_index(inplace=True, drop =True)

# val_resize2.fillna('',inplace=True)

# val_resize2['count'] = np.sum(val_resize2.iloc[:,1:]!='', axis = 1).values

# val_resize2.head()
# valid_batches_resize = DataGenerator_Resize(val_resize2[val_resize2.index.isin(defects)],batch_size= 16, shuffle=False,info=filenames )
# for i, batch in enumerate(valid_batches_resize):

#     plt.figure(figsize=(14,50))

#     for k in range(16):

#         plt.subplot(16,1, k+1)

#         img = batch[0][k,]

#         img = Image.fromarray(img.astype('uint8'))

#         img = np.array(img)

# #         print(img.shape)

#         extra = ' has defect'

#         for j in range(4):

#             msk = batch[1][k, : , : , j]

#             msk = mask2pad(msk, pad =3)

#             msk = mask2contour(msk, width =2)

#             if np.sum(msk)!=0 :

#                 extra +=' ' + str(j+1)

#             if j==0:

#                 img[msk==1,0]==235

#                 img[msk==1,1]=235

#             elif j==1:

#                 img[msk==1,1]=210

#             elif j==2:

#                 img[msk==1,2]=255

#             elif j==3:

#                 img[msk==1,0]=255

#                 img[msk==1,2]=255

#         plt.title(filenames[16*i+k] + extra)

#         plt.axis('off')

#         plt.imshow(img)

#     plt.subplots_adjust(wspace = 0.05)

#     plt.show()
test = pd.read_csv(path + 'sample_submission.csv')

test['ImageId'] = test['ImageId_ClassId'].map(lambda x: x.split('_')[0])

# test_batches = DataGenerator(test.iloc[::4],subset='test',batch_size=1)

test.head() 
from tqdm import tqdm
test.shape
# train['ImageId'] = train['ImageId_ClassId'].map(lambda x : x.split('.')[0] + '.jpg')

test2 = pd.DataFrame({'ImageId' : test['ImageId'][::4]})

test2['e1'] = test['EncodedPixels'][::4].values

test2['e2'] = test['EncodedPixels'][1::4].values

test2['e3'] = test['EncodedPixels'][2::4].values

test2['e4'] = test['EncodedPixels'][3::4].values

test2.reset_index(inplace=True, drop =True)

test2.fillna('',inplace=True)

test2['count'] = np.sum(test2.iloc[:,1:]!='', axis = 1).values
test2.shape
from multiprocessing import Pool
# test_generator = DataGenerator(test2, subset = 'test', batch_size=1)

# test_preds = model.predict_generator(test_generator,verbose=1)
# gc.collect()
def post_preprocess(preds):

#     print(preds[0][0][0])

    

    pred_masks = np.zeros((len(preds), 256,1600,4))

    for k in range(0,len(preds)):

#         mask = np.squeeze(np.round(preds[k,]))

        mask = np.squeeze(preds[k,])

        mask_1 = np.array(mask[:,:,0] > mask_bound_1, dtype=np.uint8)

        mask_1 = mask2pad(mask_1, pad=2)

        mask_1 = np.array(mask2contour(mask_1,width=3),dtype=np.uint8)

        

        mask_2 = np.array(mask[:,:,1] > mask_bound_2, dtype=np.uint8)

        mask_2 = mask2pad(mask_2, pad=2)

        mask_2 = np.array(mask2contour(mask_2,width=3), dtype=np.uint8)

        

        mask_3 = np.array(mask[:,:,2] > mask_bound_3, dtype=np.uint8)

        mask_3 = mask2pad(mask_3, pad=2)

        mask_3 = np.array(mask2contour(mask_3,width=3), dtype=np.uint8)

        

        

        mask_4 = np.array(mask[:,:,3] > mask_bound_4, dtype=np.uint8)

        mask_4 = mask2pad(mask_4, pad=2)

        mask_4 = np.array(mask2contour(mask_4,width=3), dtype=np.uint8)

        mask_re = np.stack([mask_1,mask_2,mask_3,mask_4], axis =2)

#         mask  = np.array(mask > mask_bound, dtype=np.uint8)

        mask_re = cv2.resize(mask_re, (1600,256))

        mask_re = masks_reduce(mask_re)

        pred_masks[k] = mask_re

#         gc.collect()

    return pred_masks
def paralleize_numpy(preds, func, cores = 6):

#     print("Bound : {}, Threshold : {}".format(mask_bound, mask_threshold))

    np_split = np.array_split(preds, cores )

    pool = Pool(cores)

    res_np = np.concatenate(pool.map(func, np_split))

    pool.close()

    pool.join()

    return res_np
os.cpu_count()





#         mask_1 = np.array(mask[:,:,0] > mask_bound_1, dtype=np.uint8)

#         mask_2 = np.array(mask[:,:,1] > mask_bound_2, dtype=np.uint8)

#         mask_3 = np.array(mask[:,:,2] > mask_bound_2, dtype=np.uint8)

#         mask_4 = np.array(mask[:,:,3] > mask_bound_3, dtype=np.uint8)

#         mask_re = np.stack([mask_1,mask_2,mask_3,mask_4])
test_df_resize = []

for i in tqdm(range(0,test2.shape[0], 30)):

    batch_idx = list(range(i, min(test2.shape[0] , i+30)))

    test_generator = DataGenerator(

        test2.iloc[batch_idx], subset = 'test', batch_size = 1)

    test_preds =model.predict_generator(test_generator, verbose = 1)

    test_preds_pp = paralleize_numpy(test_preds, post_preprocess, cores=2).astype(np.int32)

    

    for j, b in tqdm(enumerate(batch_idx)):

        filename = test2['ImageId'].iloc[b]

        image_df = test[test['ImageId'] == filename].copy()

        pred_rles = build_rles(test_preds_pp[j])

        image_df['EncodedPixels'] = pred_rles

        test_df_resize.append(image_df)

    gc.collect()
# mask_threshold = 1024

# mask_bound = 0.5

# E1 - 0, E2 - 0 E3 - 572 E4 -127

# mask_threshold = 256

# mask_bound = 0.5

# E1 - 0, E2 - 0 E3 - 680 E4 -150

# mask_threshold = 0

# mask_bound = 0.5

#E1 - 0, E2 - 0 E3 - 745 E4 -162



test_df_resize = pd.concat(test_df_resize)

print(test_df_resize.shape)

test_df_resize.head()

test_df_resize[['ImageId_ClassId', 'EncodedPixels']].to_csv('submission.csv', index=False)
# mask_threshold = 256

# mask_bound = 0.5
# test_df_resize2 = []

# for i in tqdm(range(0,test2.shape[0], 30)):

#     batch_idx = list(range(i, min(test2.shape[0] , i+30)))

#     test_generator = DataGenerator(

#         test2.iloc[batch_idx], subset = 'test', batch_size = 1)

#     test_preds =model.predict_generator(test_generator, verbose = 1)

#     test_preds_pp = paralleize_numpy(test_preds, post_preprocess, cores=2).astype(np.int32)

    

#     for j, b in tqdm(enumerate(batch_idx)):

#         filename = test2['ImageId'].iloc[b]

#         image_df = test[test['ImageId'] == filename].copy()

#         pred_rles = build_rles(test_preds_pp[j])

#         image_df['EncodedPixels'] = pred_rles

#         test_df_resize2.append(image_df)

#     gc.collect()
# mask_threshold = 0

# mask_bound = 0.5

# test_df_resize3 = []

# for i in tqdm(range(0,test2.shape[0], 30)):

#     batch_idx = list(range(i, min(test2.shape[0] , i+30)))

#     test_generator = DataGenerator(

#         test2.iloc[batch_idx], subset = 'test', batch_size = 1)

#     test_preds =model.predict_generator(test_generator, verbose = 1)

#     test_preds_pp = paralleize_numpy(test_preds, post_preprocess, cores=2).astype(np.int32)

    

#     for j, b in tqdm(enumerate(batch_idx)):

#         filename = test2['ImageId'].iloc[b]

#         image_df = test[test['ImageId'] == filename].copy()

#         pred_rles = build_rles(test_preds_pp[j])

#         image_df['EncodedPixels'] = pred_rles

#         test_df_resize3.append(image_df)

#     gc.collect()




# test_df_resize = pd.concat(test_df_resize)

# print(test_df_resize.shape)

# test_df_resize.head()

# test_df_resize[['ImageId_ClassId', 'EncodedPixels']].to_csv('submission.csv', index=False)
# test2_1 = pd.DataFrame({'ImageId' : test_df_resize['ImageId'][::4]})

# test2_1['ImageId'] = test_df_resize['ImageId_ClassId'].map(lambda x : x.split('.')[0] + '.jpg')



# test2_1['e1'] = test_df_resize['EncodedPixels'][::4].values

# test2_1['e2'] = test_df_resize['EncodedPixels'][1::4].values

# test2_1['e3'] = test_df_resize['EncodedPixels'][2::4].values

# test2_1['e4'] = test_df_resize['EncodedPixels'][3::4].values

# test2_1.reset_index(inplace=True, drop =True)

# test2_1.fillna('',inplace=True)

# test2_1['count'] = np.sum(test2_1.iloc[:,1:]!='', axis = 1).values
# print(

#     "E1 - {}, E2 - {} E3 - {} E4 -{}".format(

#         test2_1[test2_1['e1']!=''].shape[0],

#         test2_1[test2_1['e2']!=''].shape[0],

#         test2_1[test2_1['e3']!=''].shape[0],

#         test2_1[test2_1['e4']!=''].shape[0],)

# )
# test_df_resize2 = pd.concat(test_df_resize2)

# print(test_df_resize2.shape)

# test_df_resize2.head()
# test2_2 = pd.DataFrame({'ImageId' : test_df_resize2['ImageId'][::4]})

# test2_2['ImageId'] = test_df_resize2['ImageId_ClassId'].map(lambda x : x.split('.')[0] + '.jpg')



# test2_2['e1'] = test_df_resize2['EncodedPixels'][::4].values

# test2_2['e2'] = test_df_resize2['EncodedPixels'][1::4].values

# test2_2['e3'] = test_df_resize2['EncodedPixels'][2::4].values

# test2_2['e4'] = test_df_resize2['EncodedPixels'][3::4].values

# test2_2.reset_index(inplace=True, drop =True)

# test2_2.fillna('',inplace=True)

# test2_2['count'] = np.sum(test2_2.iloc[:,1:]!='', axis = 1).values
# print(

#     "E1 - {}, E2 - {} E3 - {} E4 -{}".format(

#         test2_2[test2_2['e1']!=''].shape[0],

#         test2_2[test2_2['e2']!=''].shape[0],

#         test2_2[test2_2['e3']!=''].shape[0],

#         test2_2[test2_2['e4']!=''].shape[0],)

# )
# test_df_resize3 = pd.concat(test_df_resize3)

# print(test_df_resize3.shape)

# test_df_resize3.head()
# test2_3 = pd.DataFrame({'ImageId' : test_df_resize3['ImageId'][::4]})

# test2_3['ImageId'] = test_df_resize3['ImageId_ClassId'].map(lambda x : x.split('.')[0] + '.jpg')



# test2_3['e1'] = test_df_resize3['EncodedPixels'][::4].values

# test2_3['e2'] = test_df_resize3['EncodedPixels'][1::4].values

# test2_3['e3'] = test_df_resize3['EncodedPixels'][2::4].values

# test2_3['e4'] = test_df_resize3['EncodedPixels'][3::4].values

# test2_3.reset_index(inplace=True, drop =True)

# test2_3.fillna('',inplace=True)

# test2_3['count'] = np.sum(test2_3.iloc[:,1:]!='', axis = 1).values
# print(

#     "E1 - {}, E2 - {} E3 - {} E4 -{}".format(

#         test2_3[test2_3['e1']!=''].shape[0],

#         test2_3[test2_3['e2']!=''].shape[0],

#         test2_3[test2_3['e3']!=''].shape[0],

#         test2_3[test2_3['e4']!=''].shape[0],)

# )
## version 10

# test_df_resize = []

# for i in range(0,test2.shape[0], 300):

#     batch_idx = list(range(i, min(test2.shape[0] , i+300)))

#     test_generator = DataGenerator(

#         test2.iloc[batch_idx], subset = 'test', batch_size = 1)

#     test_preds =model.predict_generator(test_generator, verbose = 1)

    

#     for j, b in tqdm(enumerate(batch_idx)):

#         filename = test2['ImageId'].iloc[b]

#         image_df = test[test['ImageId'] == filename].copy()

        

#         pred_masks = np.squeeze(np.round(test_preds[j,])).astype(np.uint8)

# #         pred_masks = test_preds[j, ].round().astype(int)

#         pred_masks_re = cv2.resize(pred_masks, (1600,256))

        



#         pred_rles = build_rles(pred_masks_re)

#         image_df['EncodedPixels'] = pred_rles

#         test_df_resize.append(image_df)

#     gc.collect()
# test_df_resize = pd.concat(test_df_resize)

# print(test_df_resize.shape)

# test_df_resize.head()
# test_df_resize.tail()
# final_test_df = test_df_resize.drop_duplicates(inplace=False, subset=['ImageId_ClassId'])

# test_df_resize[['ImageId_ClassId', 'EncodedPixels']].to_csv('submission.csv', index=False)
# test_df = []

# for i in range(0,test2.shape[0], 300):

#     batch_idx = list(range(i, min(test2.shape[0] , i+300)))

#     test_generator = DataGenerator(

#         test2.iloc[batch_idx], subset = 'test', batch_size = 1)

#     test_preds =model.predict_generator(test_generator, verbose = 1)

    

#     for j, b in tqdm(enumerate(batch_idx)):

#         filename = test2['ImageId'].iloc[b]

#         image_df = test[test['ImageId'] == filename].copy()

        

#         pred_masks = np.squeeze(np.round(test_preds[j,])).astype(np.uint8)

# #         pred_masks = test_preds[j, ].round().astype(int)

#         pred_masks_re = cv2.resize(pred_masks, (1600,256))



#         pred_rles = build_rles(pred_masks_re)

#         image_df['EncodedPixels'] = pred_rles

#         test_df.append(image_df)

#     gc.collect()

        

        

        

        

#         masks = np.squeeze(np.round(test_preds[j,]))

# #         print(masks)

# #         print(aaaa)

#         masks = np.array(masks >mask_bound, dtype=np.uint8 )

#         masks = cv2.resize(masks, (1600, 256))

        

#         masks =masks_reduce(masks)

# #         print(masks.shape)

# #         print(image_df)

        

# #         for idx in range(masks.shape[-1]):

# # #             print(image_df['ImageId_ClassId'][idx])

# # #             print(masks[:,:,idx])

# # #             print('=====')

# # #             image_df['EncodedPixels'][idx] = 

# # #             image_df = image_df.append(pd.DataFrame([[image_df['ImageId_ClassId'][idx], mask2rle(masks[:,:,idx])]], 

# # #                                                       columns = ["ImageId_ClassId", "EncodedPixels"]))

# #             pred_rles = build_rles(masks)

# #             image_df['EncodedPixels'][idx] = pred_rles

        

        

#         pred_rles = build_rles(masks)

        

#         image_df['EncodedPixels'] = pred_rles

#         test_df.append(image_df)

        

# #         pred_masks = test_preds[j, ].round().astype(int)

# #         pred_rles = build_rles(pred_masks)

        

# #         image_df['EncodedPixels'] = pred_rles

# #         test_df.append(image_df)

#     gc.collect()





# test_df = []

# for i in range(0,test.shape[0], 300):

#     batch_idx = list(range(i, min(test.shape[0] , i+300)))

#     test_generator = DataGenerator(

#         test.iloc[batch_idx], subset = 'test', batch_size = 1)

#     test_preds =model.predict_generator(test_generator, verbose = 1)

    

#     for j, b in tqdm(enumerate(batch_idx)):

#         filename = test['ImageId'].iloc[b]

#         image_df = test[test['ImageId'] == filename].copy()

        

#         pred_masks = test_preds[j, ].round().astype(int)

#         pred_rles = build_rles(pred_masks)

        

#         image_df['EncodedPixels'] = pred_rles

#         test_df.append(image_df)

#     gc.collect()
# train = pd.read_csv(os.path.join(path, 'train.csv'))

# train['ImageId'] = train['ImageId_ClassId'].map(lambda x : x.split('.')[0] + '.jpg')

# train2 = pd.DataFrame({'ImageId' : train['ImageId'][::4]})

# train2['e1'] = train['EncodedPixels'][::4].values

# train2['e2'] = train['EncodedPixels'][1::4].values

# train2['e3'] = train['EncodedPixels'][2::4].values

# train2['e4'] = train['EncodedPixels'][3::4].values

# train2.reset_index(inplace=True, drop =True)

# train2.fillna('',inplace=True)

# train2['count'] = np.sum(train2.iloc[:,1:]!='', axis = 1).values
# val_set = train2.iloc[idx:].copy()

# val_set.reset_index(inplace =)

# defects = list(val_set[val_set['e1']!=''].sample(3).index)

# defects += list(val_set[val_set['e2']!=''].sample(3).index)

# defects += list(val_set[val_set['e3']!=''].sample(7).index)

# defects += list(val_set[val_set['e4']!=''].sample(3).index)
# idx = int(0.8*len(train2))

# valid_batches = DataGenerator(train2.iloc[idx:],batch_size= 16, shuffle=False,info=val_set )
# for i, batch in enumerate(valid_batches):

#     plt.figure(figsize=(14,50))

#     for k in range(16):

#         plt.subplot(16,1, k+1)

#         img = batch[0][k,]

#         img = Image.fromarray(img.astype('uint8'))

#         img = np.array(img)

# #         print(img.shape)

#         extra = ' has defect'

#         for j in range(4):

#             msk = batch[1][k, : , : , j]

#             msk = mask2pad(msk, pad =3)

#             msk = mask2contour(msk, width =2)

#             if np.sum(msk)!=0 :

#                 extra +=' ' + str(j+1)

#             if j==0:

#                 img[msk==1,0]==235

#                 img[msk==1,1]=235

#             elif j==1:

#                 img[msk==1,1]=210

#             elif j==2:

#                 img[msk==1,2]=255

#             elif j==3:

#                 img[msk==1,0]=255

#                 img[msk==1,2]=255

# #         plt.title(val_set[16*i+k] + extra)

#         plt.axis('off')

#         plt.imshow(img)

#     plt.subplots_adjust(wspace = 0.05)

#     plt.show()
# test_df = pd.concat(test_df)

# print(test_df.shape)

# test_df.head()
# test_df
# class DataGenerator(keras.utils.Sequence):

#     def __init__(self, df, batch_size = 16 ,subset ='train', shuffle = False, preprocess = None, info={}):

#         super().__init__()

#         self.df = df

#         self.shuffle = shuffle

#         self.subset = subset

#         self.batch_size = batch_size

#         self.preprocess = preprocess

#         self.info = info

        

#         if self.subset =='train':

#             self.data_path = path +'train/'

#         elif self.subset =='valid':

#             self.data_path = path +'train/'

#         elif self.subset =='test':

#             self.data_path = path + 'test/'

#         self.on_epoch_end()

        

#     def __len__(self):

#         return int(np.floor(len(self.df) / self.batch_size))

    

#     def on_epoch_end(self):

#         self.indexes = np.arange(len(self.df))

#         if self.shuffle == True:

#             np.random.shuffle(self.indexes)

#     def __getitem__(self,index):

#         x = np.empty((self.batch_size, 256, 1600, 3), dtype=np.float32)

#         y = np.empty((self.batch_size, 256, 1600, 4), dtype=np.int8)

#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         for i,f in enumerate(self.df['ImageId'].iloc[indexes]):

#             self.info[index*self.batch_size + i] =f 

#             x[i,]=Image.open(self.data_path + f).resize((1600,256))

#             if self.subset =='train':

#                 for j in range(4):

#                     y[i,:,:,j] = rle2maskResize(self.df['e'+str(j+1)].iloc[indexes[i]])

#         if self.preprocess !=None : x= self.preprocess(x)

#         if self.subset == 'train' : return x,y

#         else: return x
# def rle2maskResize(rle):

#     # CONVERT RLE TO MASK 

#     if (pd.isnull(rle))|(rle==''): 

#         return np.zeros((256,1600) ,dtype=np.uint8)

    

#     height= 256

#     width = 1600

#     mask= np.zeros( width*height ,dtype=np.uint8)



#     array = np.asarray([int(x) for x in rle.split()])

#     starts = array[0::2]-1

#     lengths = array[1::2]    

#     for index, start in enumerate(starts):

#         mask[int(start):int(start+lengths[index])] = 1

    

#     return mask.reshape( (height,width), order='F' )[::1,::1]



# def mask2contour(mask, width=3):

#     # CONVERT MASK TO ITS CONTOUR

#     w = mask.shape[1]

#     h = mask.shape[0]

#     mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)

#     mask2 = np.logical_xor(mask,mask2)

#     mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)

#     mask3 = np.logical_xor(mask,mask3)

#     return np.logical_or(mask2,mask3) 



# def mask2pad(mask, pad=2):

#     # ENLARGE MASK TO INCLUDE MORE SPACE AROUND DEFECT

#     w = mask.shape[1]

#     h = mask.shape[0]

    

#     # MASK UP

#     for k in range(1,pad,2):

#         temp = np.concatenate([mask[k:,:],np.zeros((k,w))],axis=0)

#         mask = np.logical_or(mask,temp)

#     # MASK DOWN

#     for k in range(1,pad,2):

#         temp = np.concatenate([np.zeros((k,w)),mask[:-k,:]],axis=0)

#         mask = np.logical_or(mask,temp)

#     # MASK LEFT

#     for k in range(1,pad,2):

#         temp = np.concatenate([mask[:,k:],np.zeros((h,k))],axis=1)

#         mask = np.logical_or(mask,temp)

#     # MASK RIGHT

#     for k in range(1,pad,2):

#         temp = np.concatenate([np.zeros((h,k)),mask[:,:-k]],axis=1)

#         mask = np.logical_or(mask,temp)

    

#     return mask 
# test_batches = 
# for i,batch in enumerate(valid_batches):

#     plt.figure(figsize=(20,36))

#     for k in range(16):

#         plt.subplot(16,2,2*k+1)

#         img = batch[0][k,]

#         img = Image.fromarray(img.astype('uint8'))

#         img = np.array(img)

#         dft = 0

#         extra = '  has defect '

#         for j in range(4):

#             msk = batch[1][k,:,:,j]

#             if np.sum(msk)!=0: 

#                 dft=j+1

#                 extra += ' '+str(j+1)

#             msk = mask2pad(msk,pad=2)

#             msk = mask2contour(msk,width=3)

#             if j==0: # yellow

#                 img[msk==1,0] = 235 

#                 img[msk==1,1] = 235

#             elif j==1: img[msk==1,1] = 210 # green

#             elif j==2: img[msk==1,2] = 255 # blue

#             elif j==3: # magenta

#                 img[msk==1,0] = 255

#                 img[msk==1,2] = 255

#         if extra=='  has defect ': extra =''

#         plt.title('Train '+train2.iloc[16*i+k,0]+extra)

#         plt.axis('off') 

#         plt.imshow(img)

#         plt.subplot(16,2,2*k+2) 

#         if dft!=0:

#             msk = preds[16*i+k,:,:,dft-1]

#             plt.imshow(msk)

#         else:

#             plt.imshow(np.zeros((256,1600)))

#         plt.axis('off')

#         mx = np.round(np.max(msk),3)

#         plt.title('Predict Defect '+str(dft)+'  (max pixel = '+str(mx)+')')

#     plt.subplots_adjust(wspace=0.05)

#     plt.show()
# test_df = pd.concat(test_df)

# print(test_df.shape)

# test_df.head()
# final_test_df = test_df.drop_duplicates(inplace=False, subset=['ImageId_ClassId'])

# final_test_df[['ImageId_ClassId', 'EncodedPixels']].to_csv('submission.csv', index=False)