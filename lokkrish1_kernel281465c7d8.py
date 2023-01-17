# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np, pandas as pd, os, gc

import matplotlib.pyplot as plt, time

from PIL import Image 

import warnings

warnings.filterwarnings("ignore")
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
import tensorflow as tf

import keras
class DataGenerator(keras.utils.Sequence):

    def __init__(self, df, batch_size = 16, subset="train", shuffle=False, preprocess=None, info={}):

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

        X =np.empty((self.batch_size,128,800,3),dtype=np.float32)

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
from keras import backend as K

from keras.models import load_model



# https://www.kaggle.com/xhlulu/severstal-simple-keras-u-net-boilerplate



# COMPETITION METRIC

def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
model1 = load_model(r'../input/lokkrish/unet.model.14.hdf5',custom_objects={'dice_coef':dice_coef})
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
# test_path = "../input/severstal-steel-defect-detection/test_images/"



# test_list = os.listdir(test_path)



# data = []



# for fn in test_list:

#     abs_name = test_path + fn

#     a = Image.open(abs_name).resize((800,128))

#     x=np.expand_dims(a, axis=0)

#     pred = model1.predict(x)

#     for i in range(4):

#         pred_fi = pred[:,:,i+1].T.flatten()

#         pred_fi = np.where(pred_fi > 0.3, 1, 0)

#         pred_fi_id = np.where(pred_fi == 1)

#         pred_fi_id = make_testdata(pred_fi_id[0])

#         x = [fn + "_" + str(i+1), pred_fi_id]

#         data.append(x)
# columns = ['ImageId_ClassId', 'EncodedPixels']

# d = pd.DataFrame(data=data, columns=columns, dtype='str')



# def expand(x):

#     new_val = ''

#     val = x.split(' ')

#     for char in val:

#         doub = str(int(char)*2)

#         new_val = new_val+doub+' '

#         new_val[:-1]

#     return new_val
columns = ['ImageId_ClassId', 'EncodedPixels']
#d = pd.DataFrame(data=data, columns=columns, dtype='str')

#d['EncodedPixels'] = d['EncodedPixels'].apply(lambda x: expand(x) if x!='' else '')

d = pd.read_csv(r'../input/submission7/submission(4).csv')


d['EncodedPixels'] = d['EncodedPixels'].apply(lambda x:(str(x)[:-1]) if len(str(x))>0 else str(""))
d.head()
d['EncodedPixels'].iloc[0]
d['EncodedPixels'].replace('na', '', regex=True, inplace=True)
d.to_csv("submission.csv",index=False)