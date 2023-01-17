%matplotlib inline

import matplotlib.pyplot as plt

import random

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import os

from tqdm import tqdm_notebook 

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

train = train[ train['EncodedPixels'].notnull() ]

print( train.shape )

train.head()
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



def mask2rle(img):

    tmp = np.rot90( np.flipud( img ), k=3 )

    rle = []

    lastColor = 0;

    startpos = 0

    endpos = 0



    tmp = tmp.reshape(-1,1)   

    for i in range( len(tmp) ):

        if (lastColor==0) and tmp[i]>0:

            startpos = i

            lastColor = 1

        elif (lastColor==1)and(tmp[i]==0):

            endpos = i-1

            lastColor = 0

            rle.append( str(startpos)+' '+str(endpos-startpos+1) )

    return " ".join(rle)



#Calc and Plot Mean Mask

meanmask = np.zeros( (256,1600) )

for i in range( train.shape[0] ):

    meanmask += rle2mask( train['EncodedPixels'].iloc[i], (256,1600,3) ).astype(np.float64)

meanmask /= train.shape[0]



plt.imshow(meanmask)
tmp = np.copy( meanmask )

tmp[tmp<np.mean(tmp)] = 0

tmp[tmp>0] = 1

plt.imshow(tmp)
def dice_coef(y_true, y_pred, smooth=1e-9  ):

    y_true_f = y_true.reshape(-1,1)

    y_pred_f = y_pred.reshape(-1,1)

    intersection = np.sum( y_true_f * y_pred_f )

    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth) 



train = pd.read_csv('../input/train.csv')
tmp = np.copy( meanmask )

tmp[tmp<np.mean(meanmask)] = 0

tmp[tmp>0] = 1



dice=[]

for i in tqdm_notebook(range( train.shape[0] )):

    if train['EncodedPixels'].iloc[i] in [np.nan]:

        dice.append( 0. )

    else:

        mask = rle2mask( train['EncodedPixels'].iloc[i], (256,1600,3) ).astype(np.float64)

        dice.append( dice_coef( mask, tmp ) )



print( 'Dice on train:', np.mean(dice) )
tmp = np.copy( meanmask )

tmp[tmp<np.mean(meanmask)] = 0

tmp[tmp>0] = 1



meanmask_rle = mask2rle(tmp)

meanmask_rle
sub = pd.read_csv( '../input/sample_submission.csv' )

sub['EncodedPixels'] = meanmask_rle

sub.to_csv('submission.csv', index=False)

sub.head(4)