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
import pathlib
import pandas as pd
path = pathlib.Path("../input/flats")
neg = get_image_files(path/"negative_samples")
true = get_image_files(path/"true_samples")
df = pd.DataFrame({"name": neg+true, 
                   "label":[0]*len(neg)+[1]*len(true)})
data = ImageDataBunch.from_df(".", df, ds_tfms=get_transforms(), size=224, bs=8
                                ).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(6)
learn.unfreeze()
learn.fit_one_cycle(6, max_lr=slice(1e-7,1e-5))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
learn.save("model")
!ls models
