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
path = Config.data_path()/'planet'

path.mkdir(exist_ok=True)

path
pd.read_csv(path/'sample_submission_v2.csv').head(5)
an_image_path = os.listdir(path / 'train-tif-v2')[1]

an_image_path
an_image_path
from PIL import Image

Image.open(path / 'train-tif-v2' / 'train_0.tif')
from fastai.vision import *
df_tags = pd.read_csv(path/'train_v2.csv')

df_tags.head()
df_tags['tags'].value_counts() / len(df_tags)
df_tags['tags'].str.split()
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
np.random.seed(42)

data = (ImageList.from_folder(path)

        .split_by_folder(train='train', valid='valid')

        .label_from_folder()

        .transform(get_transforms(do_flip=False),size=224)

        .databunch()

        .normalize() )