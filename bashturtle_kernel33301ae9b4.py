# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("/tmp/input/train"))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline

from fastai import *

from fastai.vision import *

path_images="/tmp/input/train/"
path_images
fnames = get_image_files(path_images)

fnames
np.random.seed(2)

pat = re.compile(r'/([a-z]+).\d+.jpg$')
pat
bs=64

data = ImageDataBunch.from_name_re(path_images, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs, num_workers=0

                                  ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))

print(data.classes)

len(data.classes),data.c
learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(5)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))