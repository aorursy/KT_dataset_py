import os

import gc

import cv2

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



from joblib import Parallel, delayed



from tqdm.notebook import tqdm

%matplotlib inline
!mkdir train

!mkdir test
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test  = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')



raw= pd.concat( (train,test), sort=False )

raw['sex'] = raw['sex'].fillna('na')

raw['age_approx'] = raw['age_approx'].fillna(0)

raw['anatom_site_general_challenge'] = raw['anatom_site_general_challenge'].fillna('na')



raw['sex'] = pd.factorize( raw['sex'] )[0]

raw['age_approx'] = pd.factorize( raw['age_approx'] )[0]

raw['anatom_site_general_challenge'] = pd.factorize( raw['anatom_site_general_challenge'] )[0]

raw['diagnosis'] = pd.factorize( raw['diagnosis'] )[0]

raw['benign_malignant'] = pd.factorize( raw['benign_malignant'] )[0]



for f in raw.columns[2:-1]:

    raw[f] = raw[f].astype( np.int8 )



train = raw.loc[ raw.target.notnull() ].copy()

test  = raw.loc[ raw.target.isnull() ].copy()





train['target'] = train['target'].astype( np.int8 )



del raw

print(train.shape)

print(test.shape)

train.head()
test.head()
def resize_image(fname, var0, var1, var2, var3, var4, fold='train/' ):



    img = cv2.imread( '../input/siim-isic-melanoma-classification/jpeg/'+fold+'{}.jpg'.format(fname) )

    

    img = cv2.resize( img , (256,256), interpolation = cv2.INTER_AREA )

    

    name = fold+fname+'_'+str(var0)+'_'+str(var1)+'_'+str(var2)+'_'+str(var3)+'_'+str(var4)+'.jpg'

    

    cv2.imwrite( name , img , [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    

    return 
Parallel(n_jobs=6)(delayed(resize_image)(

    

    fname = train.image_name.values[i],

    var0 = train.sex.values[i],

    var1 = train.age_approx.values[i],

    var2 = train.anatom_site_general_challenge.values[i],

    var3 = train.diagnosis.values[i],

    var4 = train.target.values[i],

    fold='train/'



) for i in tqdm(range(train.shape[0])))
Parallel(n_jobs=6)(delayed(resize_image)(

    

    fname = test.image_name.values[i],

    var0 = test.sex.values[i],

    var1 = test.age_approx.values[i],

    var2 = test.anatom_site_general_challenge.values[i],

    var3 = test.diagnosis.values[i],

    var4 = test.target.values[i],

    fold='test/'

    

) for i in tqdm(range(test.shape[0])))