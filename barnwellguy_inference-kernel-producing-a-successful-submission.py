import numpy as np, pandas as pd, os, gc

import matplotlib.pyplot as plt, time

from PIL import Image 

import warnings

warnings.filterwarnings("ignore")
path      = '../input/severstal-steel-defect-detection/'

modelFile = '../input/simplistic-unet-for-metal-defect-segmentation/modelend.hdf5'
print(os.listdir(path))

print(os.path.isfile(modelFile))
os.listdir('../working/')
# https://www.kaggle.com/ateplyuk/pytorch-starter-u-net-resnet

# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import keras



class DataGenerator(keras.utils.Sequence):

    def __init__(self, df, batch_size = 16, subset="train", shuffle=False, preprocess=None, info={}, viewTest=False):

        super().__init__()

        self.df = df

        self.shuffle = shuffle

        self.subset = subset

        self.batch_size = batch_size

        self.preprocess = preprocess

        self.info = info

        self.viewTest = viewTest

        

        if self.subset == "train":

            self.data_path = path + 'train_images/'

        elif self.subset == "test":

            self.data_path = path + 'test_images/'

        self.on_epoch_end()



    def __len__(self):

        return int(np.ceil(len(self.df)*1. / self.batch_size))

    

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.df))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)

    

    def __getitem__(self, index): 

                

        begin = index*self.batch_size

        end   = min((index+1)*self.batch_size,len(self.indexes))

        bsize = end - begin

        indexes = self.indexes[begin:end]

        

        

        

        X = np.empty((bsize,256,1600,3),dtype=np.float32)

        if self.subset == 'train' or self.viewTest: 

            y = np.empty((bsize,256,1600,4),dtype=np.int8)

        

        for i,f in enumerate(self.df['ImageId'].iloc[indexes]):

            self.info[index*self.batch_size+i]=f

            X[i,] = Image.open(self.data_path + f)

            if self.subset == 'train' or self.viewTest: 

                for j in range(4):

                    y[i,:,:,j] = rle2mask(self.df['e'+str(j+1)].iloc[indexes[i]])

        if self.preprocess!=None: X = self.preprocess(X)

            

        assert not np.any(np.isnan(X))

        if self.subset == 'train' or self.viewTest: 

            assert not np.any(np.isnan(y))

            return X, y

        else: return X
# https://www.kaggle.com/titericz/building-and-visualizing-masks

def rle2mask(rle):

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

    return mask.reshape( (height,width), order='F' )[::1,::1]



def mask2rle(mask):

    startEnd = np.diff(np.concatenate(([0],mask.T.flatten(),[0])))

    starts   = np.where(startEnd== 1)[0]

    if len(starts) == 0:

        return ''

    ends     = np.where(startEnd==-1)[0]

    length   = ends - starts

    starts  += 1    # it seems the data set pixel index starts at 1

    return ' '.join(['{} {}'.format(s,l) for s,l in zip(starts,length)])



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
from keras import backend as K

# https://www.kaggle.com/xhlulu/severstal-simple-keras-u-net-boilerplate



# COMPETITION METRIC

def dice_coef(y_true, y_pred, smooth=1.0):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def neg_dice_coef(y_true, y_pred, smooth=1.0):

    return - dice_coef(y_true, y_pred, smooth=smooth)
#from segmentation_models import Unet

#from segmentation_models.backbones import get_preprocessing



# LOAD UNET WITH PRETRAINING FROM IMAGENET

#preprocess = get_preprocessing('resnet34') # for resnet, img = (img-110.0)/1.0

preprocess = lambda x:x
# LOAD MODEL

from keras.models import load_model

model = load_model(modelFile,custom_objects={'dice_coef':dice_coef})
# PREDICT 1 BATCH TEST DATASET

test            = pd.read_csv(path + 'sample_submission.csv')

test['ImageId'] = test['ImageId_ClassId'].map(lambda x: x.split('_')[0])
outputs = []



#the test set run after submission is different from the test set we download. The count of picutres are perhaps also different.

blockLength = 2000

assert blockLength % 4 == 0



for begin in np.arange(0,len(test),blockLength):

    test_batches = DataGenerator(test.iloc[begin:begin+blockLength:4],subset='test',batch_size=16,preprocess=preprocess)

    test_preds   = model.predict_generator(test_batches,steps=None,verbose=1)

    for i in range(len(test_preds)):

        for t in range(4):

            thresholded                          = np.zeros_like(test_preds[i,:,:,t])

            thresholded[test_preds[i,:,:,t]>0.5] = 1

            outputs.append(mask2rle(thresholded))

            if i < 100:

                print(i,t,np.sum(thresholded))

                

assert len(test) == len(outputs)

                

test['EncodedPixels'] = outputs     
test
test.drop(columns='ImageId').to_csv('submission.csv',index=False)
test2 = pd.DataFrame({'ImageId':test['ImageId'][::4]})

test2['e1'] = test['EncodedPixels'][::4].values

test2['e2'] = test['EncodedPixels'][1::4].values

test2['e3'] = test['EncodedPixels'][2::4].values

test2['e4'] = test['EncodedPixels'][3::4].values
test2 = test2.sample(20,random_state=0).copy().reset_index(drop=True)

test2.fillna('',inplace=True); 


filenames = {}

# defects  = list(train2[train2['e1']!=''].sample(3).index)

# defects += list(train2[train2['e2']!=''].sample(3).index)

# defects += list(train2[train2['e3']!=''].sample(7).index)

# defects += list(train2[train2['e4']!=''].sample(3).index)



# defects  = list(train2[train2['e4']!=''].sample(20).index)

# defects += list(train2[train2['e2']!=''].sample(3).index)

# defects += list(train2[train2['e3']!=''].sample(7).index)

# defects += list(train2[train2['e4']!=''].sample(3).index)



# DATA GENERATOR

train_batches = DataGenerator(test2,shuffle=False,info=filenames,subset='test',viewTest=True)

print('Images and masks from our Data Generator')

print('KEY: yellow=defect1, green=defect2, blue=defect3, magenta=defect4')



# DISPLAY IMAGES WITH DEFECTS

for i,batch in enumerate(train_batches):

    plt.figure(figsize=(14,50)) #20,18

    for k in range(len(batch[0])):

        plt.subplot(16,1,k+1)

        img = batch[0][k,]

        img = Image.fromarray(img.astype('uint8'))

        img = np.array(img)

        extra = '  has defect'

        for j in range(4):

            msk = batch[1][k,:,:,j]

            #msk = mask2pad(msk,pad=3)

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