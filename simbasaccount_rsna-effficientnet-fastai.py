# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

from glob import *

import matplotlib.pyplot as plt

from PIL import Image

from fastai.vision import *

# Any results you write to the current directory are saved as output.

from fastai.callbacks import *
from pathlib import Path

Path.ls = lambda x: list(x.iterdir())
path_train = Path('/kaggle/input/')
(path_train/'rsna-train-stage-1-images-png-224x/').ls()
train_files = sorted(glob("../input/rsna-train-stage-1-images-png-224x/stage_1_train_png_224x/*.png"))
len(train_files)
train = pd.read_csv(os.path.join('/kaggle/input/rsna-intracranial-hemorrhage-detection', 'stage_1_train.csv'))

train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)

train = train[['Image', 'Diagnosis', 'Label']]

train.drop_duplicates(inplace=True)

train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()

train['Image'] = 'ID_' + train['Image']

train.head()
dir_csv = '../input/rsna-intracranial-hemorrhage-detection'

dir_train_img = '../input/rsna-train-stage-1-images-png-224x/stage_1_train_png_224x'

dir_test_img = '../input/rsna-test-stage-1-images-png-224x/stage_1_test_png_224x'
png = glob(os.path.join(dir_train_img, '*.png'))

png = [os.path.basename(png)[:-4] for png in png]

png = np.array(png)



train = train[train['Image'].isin(png)]

train.to_csv('train.csv', index=False)
src = (ImageList.from_df(df = train,

                          path = '/kaggle/input/rsna-train-stage-1-images-png-224x/stage_1_train_png_224x',

                          cols = 'Image',

                          suffix='.png')

        .split_by_rand_pct(0.1)

        .label_from_df(cols = ['any', 'epidural', 'intraparenchymal','intraventricular','subarachnoid','subdural'])

       )
tfms = get_transforms(do_flip = True)
data = (src

       .databunch(bs = 64, num_workers= 4)

       .normalize(imagenet_stats))
data.show_batch()
acc_02 = partial(accuracy_thresh, thresh=0.2)

f_score = partial(fbeta, thresh=0.2)
data.c
!pip install efficientnet-pytorch

from efficientnet_pytorch import EfficientNet
model_effnetb2 =  EfficientNet.from_pretrained('efficientnet-b2', num_classes=data.c)
learn = Learner(data,

                model_effnetb2,

                metrics = [acc_02, f_score],

                callback_fns=[partial(EarlyStoppingCallback, monitor='acc_02', min_delta=0.01, patience=3)], path = '/kaggle/working', model_dir = '/kaggle/working',

                

                wd=1e-3)




learn = learn.split([learn.model._conv_stem,learn.model._blocks,learn.model._conv_head])



#learn = cnn_learner(data, base_arch=models.resnet50, metrics = [acc_02, f_score], callback_fns=[partial(EarlyStoppingCallback, monitor='acc_02', min_delta=0.01, patience=3)], path = '/kaggle/working', model_dir = '/kaggle/working' )
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, max_lr = 2e-3)
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, max_lr = slice(2e-5, 7e-5), wd = 1e-1)
learn.plot_losses()
learn.export('trained_1.pkl')