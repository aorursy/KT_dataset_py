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
%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *

np.random.seed(42)
path_img = '../input/chest-xrays-radiopaedia/images/images'
fnames = get_image_files(path_img)
fnames[:3]
img_f = fnames[0]
img = open_image(img_f)
img.show(figsize=(5,5))
img_size = np.array(img.shape[1:])
img_size,img.data
size = img_size//2
bs =2
path = '../input/chest-xrays-radiopaedia'
data = ImageDataBunch.from_csv(path, folder="/images/images", valid_pct=0.2, csv_labels='radiopaedia_metadata.csv',
         ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data = (data.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))


data.show_batch(2, figsize=(10,7))

data.show_batch(2, figsize=(10,7), ds_type=DatasetType.Valid)


