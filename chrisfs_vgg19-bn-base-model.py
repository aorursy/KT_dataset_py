from fastai import *

from fastai.vision import *

import pandas as pd

import matplotlib.pyplot as plt

import pandas as pd 

import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import matplotlib.pyplot as plt

import imagehash

import psutil



from PIL import Image

from joblib import Parallel, delayed



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from IPython import display

import time

base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')

train_dir = os.path.join(base_image_dir,'train_images/')

df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))

df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))

df = df.drop(columns=['id_code'])

df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe

df.head()
bs = 32 #smaller batch size is better for training, but may take longer

sz=140
tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=0.4,max_warp=0,max_zoom=1.4,max_lighting=0.4,p_lighting=0.5)

src = (ImageList.from_df(df=df,path='./',cols='path') #get dataset from dataset

        .split_by_rand_pct(0.2) #Splitting the dataset

        .label_from_df(cols='diagnosis') #obtain labels from the level column

      )

data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') #Data augmentation

        .databunch(bs=bs,num_workers=4) #DataBunch

        .normalize(imagenet_stats) #Normalize     

       )
from sklearn.metrics import cohen_kappa_score

def quadratic_kappa(y_hat, y):

    return torch.tensor(cohen_kappa_score(torch.argmax(y_hat,1), y, weights='quadratic'),device='cuda:0')
if not os.path.exists('/tmp/.cache/torch/checkpoints/'):

        os.makedirs('/tmp/.cache/torch/checkpoints/')

!cp '../input/vgg19bn/vgg19_bn.pth' '/tmp/.cache/torch/checkpoints/vgg19_bn-c79401a0.pth'
learn= cnn_learner(data, base_arch=models.vgg19_bn,  metrics = [accuracy,quadratic_kappa])


learn.fit_one_cycle(7)

# learn.fit_one_cycle(4, max_lr=1e-2)

#learn.recorder.plot_losses()

learn.lr_find()

learn.recorder.plot()
#progressive resizing

learn.data = data = (

    src.transform(get_transforms(tfms),size=255)

    .databunch(bs=bs,num_workers=4)

    .normalize()

)



learn.freeze()
learn.lr_find()

learn.recorder.plot()



learn.unfreeze()

learn.fit_one_cycle(9,slice(1e-4,1e-3))
sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

learn.data.add_test(ImageList.from_df(sample_df,'../input/aptos2019-blindness-detection/',folder='test_images',suffix='.png'))

preds,y = learn.get_preds(DatasetType.Test)
sample_df.diagnosis = preds.argmax(1)

sample_df.head()



sample_df.to_csv('submission.csv',index=False)