# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install fastai --upgrade
from fastai.vision.all import *
path = Path('../input/drive-digital-retinal-images-for-vessel-extraction/DRIVE')
path_training = Path('../input/drive-digital-retinal-images-for-vessel-extraction/DRIVE/training')
images = get_image_files(path_training/'images')
img = PILImage.create(images[0])
img
codes = ['0','1']
codes
def label_func(fn):
    return path_training/'1st_manual'/f"{fn.stem.split('_')[0]}_manual1.gif"
tfms = [IntToFloatTensor(div_mask=255),Resize(224),Zoom(max_zoom=2.1,p=0.5),Normalize.from_stats(*imagenet_stats)]



db = DataBlock(blocks=(ImageBlock(),MaskBlock()),
               batch_tfms=[IntToFloatTensor(div_mask=255),Resize(224),Zoom(max_zoom=2.1,p=0.5),Normalize.from_stats(*imagenet_stats)],
               get_items=get_image_files,
               get_y=label_func)


dls = db.dataloaders(path_training/'images',bs = 4)
dls.show_batch(max_n=6)
learn = unet_learner(dls,resnet34,metrics=Dice(),n_out=2)

learn.fine_tune(20)
learn.lr_find()
learn.fit_one_cycle(4, lr_max = 1e-5)
learn.show_results(max_n=2, figsize=(16,16))

learn = unet_learner(dls,xresnet34,metrics=Dice(),n_out=2)
learn.fine_tune(8)
learn.lr_find()
learn.fit_one_cycle(4, lr_max = 3e-4)
learn.show_results(max_n=4, figsize=(16,16))
