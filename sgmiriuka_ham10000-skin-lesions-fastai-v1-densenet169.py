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
from fastai import *

from fastai.vision import *

from fastai.callbacks.hooks import *

from fastai.callbacks import *

from torchvision.models import *

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

from glob import glob

path = Path("../input")

labels = pd.read_csv('../input/HAM10000_metadata.csv', sep=',')

labels.head()



imageid = {os.path.splitext(os.path.basename(x))[0]: x

                     for x in glob(os.path.join(path, '*', '*.jpg'))}



labels['path'] = labels['image_id'].map(imageid.get)

labels['path'] = labels['path'].str[9:]

labels.sample(5)
unique_lesion = labels['lesion_id'].value_counts()
sns.set(style="white", palette="pastel", color_codes=True)

f, (ax1) = plt.subplots(1, 1, figsize=(6, 6))

sns.countplot(x=unique_lesion, palette="Blues", ax=ax1)

plt.title('Number of images per lesion')

plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

sns.countplot(x=labels['sex'], palette="Blues", ax=ax1)

sns.boxplot(x=labels['age'], orient='v', ax=ax2)

plt.show()
f, ax1 = plt.subplots(1,1, figsize=(10,5))

sns.boxplot(x=labels['dx'], y=labels['age'], hue=labels['sex'], ax=ax1)

plt.title('Diagnosis by gender')

plt.show()
f, (ax1) = plt.subplots(1, 1, figsize=(8, 5), sharex=True)

sns.countplot(x=labels['dx_type'], palette="Greens", ax=ax1)

plt.show()
f, (ax1) = plt.subplots(1, 1, figsize=(8, 5), sharex=True)

sns.countplot(x=labels['localization'], palette="Reds", ax=ax1)

plt.xticks(rotation=90)

plt.show()
tfms = get_transforms(do_flip=True, 

                      flip_vert=True,

                      max_zoom=1.1,

                      max_warp=0.2, 

                      p_affine=0.5,

                      xtra_tfms=[rotate(degrees=(-45,45),p=.1),

                                brightness(change=(0.35,0.65),p=.5),

                                contrast(scale=(0.8,1.2),p=.5),

                                dihedral(p=1)])
np.random.seed(21)

data = ImageDataBunch.from_df(path='../input/', df=labels,

                              ds_tfms=tfms, size=224,bs=16,

                               valid_pct=0.2, fn_col='path', 

                              label_col='dx'

                              ).normalize(imagenet_stats)
data.show_batch(rows=3)
arch=densenet121

learner = create_cnn(data, arch=arch, metrics=[accuracy],ps=.5,model_dir="/tmp/model/",

                    callback_fns=ShowGraph).to_fp16()
learner.lr_find()

learner.recorder.plot()
learner.fit(1, 1e-2)
learner.unfreeze()

learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(3, max_lr=slice(1e-5, 1e-3), wd=0.1)
learner.to_fp32()   #I think that in latest version of fastai you don't have to come back 

                    #from half_precision, but here gives me an error when running TTA.

int=learner.interpret(tta=True)
int.plot_confusion_matrix(figsize=(6,6))
int.plot_top_losses(9, figsize=(10,10), heatmap=False)