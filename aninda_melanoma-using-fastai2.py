import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import glob

fn = glob.glob('/kaggle/input/siic-isic-224x224-images/train/*.*')

len(fn)
!pip install fastai2
import pandas as pd

import numpy as np

import random

import geopandas as gpd

import rasterio

from PIL import Image

from matplotlib import pyplot as plt

%matplotlib inline

from fastai2.torch_basics import *

from fastai2.basics import *

from fastai2.data.all import *

from fastai2.callback.all import *

from fastai2.vision.all import *

from fastai2.test_utils import *

from fastai2.vision.core import *

from fastai2.metrics import *

from sklearn.metrics import roc_auc_score
train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

train.shape
train_1 = train[train['target']==1]

train_1.shape
train_0 = train[train['target']==0].sample(frac=0.018)

train_0.shape
df = pd.concat([train_0,train_1]).reset_index(drop=True)

df.shape
df.head()
#Read an image

img = Image.open('/kaggle/input/siic-isic-224x224-images/train/ISIC_0645834.png')

plt.imshow(np.asarray(img))
# CHeck shape and number of True labels

df.shape, df.target.value_counts()
df['file_name'] = df['image_name'].apply(lambda x: f"/kaggle/input/siic-isic-224x224-images/train/{x}"+".png" )
df.head()
#df['target']=df['target'].apply(lambda x: float(x))

df.head()
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3,shuffle=True,random_state=42)

X=df['file_name'].copy()

y=df['target'].copy()

fold = 0

for train_index, test_index in skf.split(X, y):

    fold+= 1

    print('In fold',fold)

    print("TRAIN LENGTH:", len(train_index), "VALIDATION LENGTH:", len(test_index))

    df[f'fold_{fold}_valid']=False

    df.loc[test_index,f'fold_{fold}_valid']=True
df.head()
test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')

test.shape
test['file_name'] = test['image_name'].apply(lambda x: f"/kaggle/input/siic-isic-224x224-images/test/{x}"+".png" )
test.head()
ss = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')

ss.shape
#roc_auc=skm_to_fastai(roc_auc_score)

roc_auc = RocAuc()

metrics = [accuracy,roc_auc]
def dataloader(fold):

    tfm=aug_transforms(do_flip=True, flip_vert=True, max_rotate=45.0, max_zoom=1.1, size=224,max_lighting=0.2, max_warp=0.4, p_affine=0.75, p_lighting=0.75, xtra_tfms=None, mode='bilinear')

    dls = ImageDataLoaders.from_df(df, fn_col='file_name',label_col='target', valid_col=f'fold_{fold}_valid',path='', folder='/', seed=42,batch_tfms = [*tfm, Normalize.from_stats(*imagenet_stats)],bs=32,num_workers=0)

    return dls
final_preds=np.zeros((ss.shape[0],ss.shape[1]))

fold = 0

for fold in range(3):

    fold+=1

    print('In fold:',fold)

    dls=dataloader(fold)

    learn = cnn_learner(dls,resnet34,metrics=metrics)

    learn.fine_tune(10)

    test_dl=learn.dls.test_dl(test)

    preds, _ = learn.tta(dl=test_dl)

    print('Prediction completed in fold: {}'.format(str(fold)))

    final_preds+=preds.numpy()

    



final_preds=final_preds/3
test['target'] = final_preds[:,1]

test.head()
ss.head()
test[['image_name', 'target']].to_csv('Sub.csv', index=False)