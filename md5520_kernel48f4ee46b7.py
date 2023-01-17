import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras import backend as K

import keras

from keras.preprocessing import image

import matplotlib.pyplot as plt



def rle2Mask(rle, image_height = 256, image_width = 1600):

    if (pd.isnull(rle)) | (rle == ''):

        return np.zeros((image_height, image_width),dtype=np.uint8)

        

    mask = np.zeros(image_width*image_height, dtype=np.uint8)

    arrays = np.asarray([int(x) for x in rle.split(' ')])

    start_pos = arrays[::2] - 1 #-1 existed since rle start from 1

    lengths = arrays[1::2]

    for start, length in zip(start_pos, lengths):

        mask[start:start + length] = 1

    return mask.reshape((image_height, image_width),order='F')



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



def rle2maskResize(rle):

    # CONVERT RLE TO MASK 

    if (pd.isnull(rle))|(rle==''): 

        return np.zeros((256,1600) ,dtype=np.uint8)

    

    height= 256

    width = 1600

    mask= np.zeros( width*height ,dtype=np.uint8)



    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]-1

    lengths = array[1::2]    

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

    

    return mask.reshape( (height,width), order='F' )



def mask2contour(mask, width=3):

    # CONVERT MASK TO ITS CONTOUR

    w = mask.shape[1]

    h = mask.shape[0]

    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)

    mask2 = np.logical_xor(mask,mask2)

    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)

    mask3 = np.logical_xor(mask,mask3)

    return np.logical_or(mask2,mask3)



def is_defect_existed(mask,threshold = 0.8):

    return np.max(mask) > threshold



def mask_integration(mask, threshold = 0.8):

    res = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    res[mask>threshold] = 1

    return res



def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)





class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, df, batch_size=32, dim=(256,1600), n_channels=3,

                 subset='train',shuffle=True,info={},preprocess=None):

        'Initialization'

        super().__init__()

        self.df = df

        self.dim = dim

        self.batch_size = batch_size

        self.n_channels = n_channels

        self.subset = subset

        self.shuffle = shuffle

        self.info = info

        self.preprocess = preprocess

        if self.subset == 'train':

            self.datapath = '../input/severstal-steel-defect-detection/train_images/'

        elif self.subset == 'test':

            self.datapath = '../input/severstal-steel-defect-detection/test_images/'

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.df) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        X = np.empty((self.batch_size, *self.dim, self.n_channels),dtype=np.float32)

        y = np.empty((self.batch_size, *self.dim, 4),dtype=np.int8)

        # Find list of IDs

        for i,file_name in enumerate(self.df['ImageId'].iloc[indexes]):

            self.info[index*self.batch_size+i]=file_name

            X[i,] = image.img_to_array(image.load_img(self.datapath + file_name, target_size=(256,1600)))

            if self.subset == 'train':

                for j in np.arange(4):

                    y[i,:,:,j] = rle2Mask(self.df['e'+str(j+1)].iloc[indexes[i]])

        if self.preprocess!=None: X = self.preprocess(X)           

        if self.subset == 'train':

            return X, y

        else:

            return X



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.df))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)







import os

import sys

seg_whl = '../input/steel-whls/segmentation_models-0.2.1-py2.py3-none-any.whl'

classifier_whl = '../input/steel-whls/image_classifiers-0.2.0-py2.py3-none-any.whl'

!pip install '../input/steel-whls/image_classifiers-0.2.0-py2.py3-none-any.whl'

!pip install '../input/steel-whls/segmentation_models-0.2.1-py2.py3-none-any.whl'





from segmentation_models import Unet

from segmentation_models.backbones import get_preprocessing
file_train = '../input/severstal-steel-defect-detection/train.csv'

file_test = '../input/severstal-steel-defect-detection/sample_submission.csv'

palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]

#Data preprocessing

df_train_raw = pd.read_csv(file_train)

df_train_raw['EncodedPixels'].fillna('',inplace=True)

df_train_raw['ImageId'] = df_train_raw['ImageId_ClassId'].map(lambda x:x.split('_')[0])

df_train = pd.DataFrame(df_train_raw['ImageId'].iloc[::4])

df_train['e1'] = df_train_raw['EncodedPixels'].iloc[::4].values

df_train['e2'] = df_train_raw['EncodedPixels'].iloc[1::4].values

df_train['e3'] = df_train_raw['EncodedPixels'].iloc[2::4].values

df_train['e4'] = df_train_raw['EncodedPixels'].iloc[3::4].values

df_train['count'] = np.sum(df_train.iloc[:,1:]!='',axis=1)

df_train = df_train[df_train['count']!=0]

df_train.reset_index(inplace=True,drop=True)

#

df_test_raw = pd.read_csv(file_test)

df_test_raw['EncodedPixels'].fillna('',inplace=True)

df_test_raw['ImageId'] = df_test_raw['ImageId_ClassId'].map(lambda x:x.split('_')[0])

df_test = pd.DataFrame(df_test_raw['ImageId'].iloc[::4])

df_test['e1'] = df_test_raw['EncodedPixels'].iloc[::4].values

df_test['e2'] = df_test_raw['EncodedPixels'].iloc[1::4].values

df_test['e3'] = df_test_raw['EncodedPixels'].iloc[2::4].values

df_test['e4'] = df_test_raw['EncodedPixels'].iloc[3::4].values

df_test['count'] = np.sum(df_test.iloc[:,1:]!='',axis=1)

df_test = df_test[df_test['count']!=0]

df_test.reset_index(inplace=True,drop=True)

#

weights_file = '../input/steel-defect/steel_defect.h5'

idx = int(len(df_train) * 0.8)

model = keras.models.load_model(weights_file,custom_objects={'dice_coef':dice_coef})

preprocess = get_preprocessing('resnet34')

intervals = [(0,len(df_test)//4),(len(df_test)//4,len(df_test)//2),(len(df_test)//2,len(df_test)//4 * 3),(len(df_test)//4 * 3,len(df_test))]

for interval in intervals:    

    test_batches = DataGenerator(df_test.iloc[interval[0]:interval[1],:],batch_size=1,subset='test',shuffle=False,preprocess=preprocess)

    preds = model.predict_generator(test_batches,verbose=1)

    for pic_index in np.arange(preds.shape[0]):

        for defect_code in np.arange(preds.shape[3]):

            mask = preds[pic_index,:,:,defect_code]

            if is_defect_existed(mask,threshold=0.8):

                mask_int = mask_integration(mask,threshold=0.8)

                rle = mask2rle(mask_int)

                df_test.iloc[interval[0]+pic_index,defect_code+1] = rle

            else:

                df_test.iloc[interval[0]+pic_index,defect_code+1] = ""

rle_list = []

for idx in np.arange(len(df_test)):

    for column_idx in np.arange(1,5):

        rle_list.append(df_test.iloc[idx,column_idx])
df_submit = pd.read_csv('../input/severstal-steel-defect-detection/sample_submission.csv')
df_submit['EncodedPixels'] = rle_list

df_submit.to_csv('submission.csv',index=False)