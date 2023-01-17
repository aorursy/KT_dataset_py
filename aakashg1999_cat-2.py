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
from fastai import *

from fastai.vision import *
bs=64
train = "/kaggle/input/dataset-pc-hg/dataset/train"

valid = "/kaggle/input/dataset-pc-hg/dataset/valid"

path= "/kaggle/input/dataset-pc-hg/dataset"

fnames = get_image_files(valid+"/hg")

fnames[:5]
np.random.seed(2)

data = ImageDataBunch.from_folder(path,no_check=True,size=224,bs=bs, num_workers=0)

data.normalize(imagenet_stats)
print(data.classes)

len(data.classes),data.c
data.show_batch(rows=3, figsize=(10,15))

learn = cnn_learner(data, models.resnet34, metrics=error_rate,model_dir = '/kaggle/working/')
#learn.lr_find()
#learn.recorder.plot()
lr=1e-3
learn.fit_one_cycle(5,slice(lr))
img = open_image(path+'/Download.png')

#img
pred_class,pred_idx,outputs = learn.predict(img)
pred_class
learn.save('testing')
learn.load('testing')
#saved in /kaggle/working and downloaded from options