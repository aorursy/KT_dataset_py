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
!pip install torch torchvision feather-format kornia pyarrow --upgrade   > /dev/null

!pip install git+https://github.com/fastai/fastai_dev                    > /dev/null
from fastai2.basics           import *

from fastai2.vision.all       import *

from fastai2.medical.imaging  import *

from fastai2.callback.tracker import *

from fastai2.callback.all     import *



np.set_printoptions(linewidth=120)

matplotlib.rcParams['image.cmap'] = 'bone'
bs = 64
path = untar_data(URLs.PETS); path
path.ls()
path_anno = path/'annotations'

path_img = path/'images'
fnames = get_image_files(path_img)

fnames[:5]

np.random.seed(2)

pat = r'/([^/]+)_\d+.jpg$'
dbunch = ImageDataBunch.from_name_re(path, fnames, pat, item_tfms=RandomResizedCrop(460, min_scale=0.75), bs=bs,

                                     batch_tfms=[*aug_transforms(size=224, max_warp=0), Normalize(*imagenet_stats)])
dbunch.show_batch(max_n=9, figsize=(10,10))
print(dbunch.vocab)

len(dbunch.vocab),dbunch.c
learn = cnn_learner(dbunch, resnet34, metrics=error_rate).to_fp16()
learn.model
learn.fit_one_cycle(4)
learn.recorder.plot_loss()
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(dbunch.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.lr_find()
learn.fit_one_cycle(2, lr_max = slice(1e-6, 1e-4))
dbunch = ImageDataBunch.from_name_re(path_img, fnames, pat, item_tfms=RandomResizedCrop(460, min_scale=0.75), bs=bs//2,

                                     batch_tfms=[*aug_transforms(size=299, max_warp=0), Normalize(*imagenet_stats)])
learn = cnn_learner(dbunch, resnet50, metrics=error_rate).to_fp16()
learn.lr_find()
learn.fit_one_cycle(8)
learn.save('stage-1-50')
learn.unfreeze()

learn.fit_one_cycle(3, lr_max=slice(1e-6,1e-4))
learn.load('stage-1-50');
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(12,12), dpi=60)