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

from fastai.metrics import error_rate
path = Path('../input/kvasir-dataset-v2/kvasir-dataset-v2')

path.ls()
bs =64



tfms = get_transforms(do_flip=True)
data= ImageDataBunch.from_folder(path,ds_tfms=tfms,bs=bs,valid_pct=0.2,no_check=True,size=128)
data.classes
data.show_batch(rows=3 , figsize=(8,8))
learn = create_cnn(data,models.resnet34,metrics=error_rate,model_dir='/tmp/models')
lr_find(learn)

learn.recorder.plot()
learn.fit_one_cycle(4)
learn.save('Kvasir-stage1')
interp = ClassificationInterpretation.from_learner(learn)

losses,idx = interp.top_losses()

len(data.valid_ds)== len(losses)==len(idx)
interp.plot_top_losses(9,figsize=(6,6))
interp.plot_confusion_matrix(figsize=(12,12),dpi=60)
interp.most_confused(min_val=2)
learn.load('Kvasir-stage1')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()
learn.fit_one_cycle(4,max_lr=slice(1e-6,1e-4))
learn.save('Kvest-stage-2')
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))