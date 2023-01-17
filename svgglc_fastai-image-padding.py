# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

   # for filename in filenames:

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import fastai

from fastai.vision import *
path = Path('../input/data/data')
train = path/'train'

test =  path/'test'

path.ls()
#tfms =get_transforms(do_flip=False,flip_vert=True,max_rotate=0, max_zoom=0.5,max_lighting=0, max_warp=0)

tfms =get_transforms()

np.random.seed(42)

data = ImageDataBunch.from_folder(path, train="train", valid_pct=0.2,

ds_tfms=tfms, size=224, resize_method=ResizeMethod.PAD,padding_mode = "zeros", num_workers=4,test =path/'test').normalize(imagenet_stats)
len(data.train_ds)
data.show_batch(1)