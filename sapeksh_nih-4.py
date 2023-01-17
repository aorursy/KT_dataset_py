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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from glob import glob

import matplotlib.pyplot as plt



%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai.vision import *



path = Path('/kaggle/input/data')

path.ls()
all_xray_df = pd.read_csv('../input/data/Data_Entry_2017.csv')

all_image_paths = {os.path.basename(x): x for x in 

                   glob(os.path.join('..', 'input/data/', 'images*', '*', '*.png'))}

print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])

all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)

all_xray_df.head()
test = pd.read_csv('../input/testing/NIH_Test_Images.csv')

test.head()
test1 = all_xray_df.merge(test,how='inner',left_on='Image Index',right_on='ImageID')

print(test1.shape)

test1.head()
train = all_xray_df.merge(test,how='outer',left_on='Image Index',right_on='ImageID', indicator=True)

train = train[train['_merge'] == 'left_only']

train.shape
cols = train.columns.tolist()

cols = cols[-1:] + cols[:-1]

train = train[cols] 

train = train[['path','Finding Labels']] 

train.head()
cols = test1.columns.tolist()

cols = cols[-1:] + cols[:-1]

test1 = test1[cols] 

test1 = test1[['path','Finding Labels']] 

test1.head()
def sampling_k_elements(group, k=1000):

    if len(group) < k:

        return group

    return group.sample(k)



train2 = train.groupby('Finding Labels').apply(sampling_k_elements).reset_index(drop=True)

train2.shape
tfms = get_transforms()



src = (ImageList.from_df(train2, '')

       .split_by_rand_pct(0.2)

       .label_from_df(label_delim='|')

       )
data = (src.transform(tfms, size=256)

        .databunch(num_workers=0).normalize(imagenet_stats))

data.show_batch(rows=2, figsize=(10,10))
from fastai.metrics import error_rate



arch = models.resnet152

acc_02 = partial(accuracy_thresh, thresh=0.2)

f_score = partial(fbeta, thresh=0.2)

learn = cnn_learner(data, arch, metrics=[acc_02, f_score])
learn.load('/kaggle/input/nih-3/final_stage-4-128-resnet152')
lr=1e-2



learn.unfreeze()



learn.fit_one_cycle(15, slice(1e-4, lr/5))
learn.save('/kaggle/working/final_stage-5-256-resnet152')
import numpy as np

from sklearn.metrics import roc_auc_score



log_preds, test_labels = learn.get_preds(ds_type=DatasetType.Valid)

roc_auc_score(test_labels, log_preds, average = 'macro')
print(set(zip(roc_auc_score(test_labels, log_preds, average = None),data.classes)))
tfms = get_transforms()



src = (ImageList.from_df(test1, '')

       .split_none()

       .label_from_df(label_delim='|')

      )



data = (src.transform(tfms, size=256)

        .databunch(num_workers=0).normalize(imagenet_stats))

    

from fastai.metrics import error_rate



arch = models.resnet152

learn = cnn_learner(data, arch)



learn.load('/kaggle/working/final_stage-5-256-resnet152')



import numpy as np

from sklearn.metrics import roc_auc_score



log_preds, test_labels = learn.get_preds(ds_type=DatasetType.Train)

print(roc_auc_score(test_labels, log_preds, average = 'macro'))

print(set(zip(roc_auc_score(test_labels, log_preds, average = None),data.classes)))
