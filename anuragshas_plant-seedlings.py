!pip install pytorchcv
!pip install fastai==1.0.47
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
from pathlib import Path

import fastai

from fastai.vision import *

from pytorchcv.model_provider import get_model as ptcv_get_model
PATH = Path('../input/train/')

np.random.seed(42)
batch_size = 128
tfms = get_transforms(flip_vert=True,max_lighting=0.5,max_rotate=360.,max_warp=0.3)
src = (ImageList.from_folder(path=PATH)

            .split_by_rand_pct(0.05)

            .label_from_folder())
data = (src.transform(tfms,size=224)

          .databunch(bs=batch_size)

           .normalize(imagenet_stats))
# data.show_batch(rows=3, figsize=(9, 9))
# data.classes, data.c
def md(f=None):

    mdl = ptcv_get_model('condensenet74_c4_g4', pretrained=True)

    mdl.features.final_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)

    return mdl
learn = cnn_learner(data, md, metrics=accuracy, path='./')
# learn.summary()
learn.freeze()
# learn.lr_find()

# learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5,slice(1e-2))
learn.unfreeze()
# learn.lr_find()

# learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5,slice(1e-6))
# interp = ClassificationInterpretation.from_learner(learn)

# interp.plot_top_losses(9, figsize=(14,14))
# interp.plot_confusion_matrix(figsize=(7, 7),dpi=90)
test = ImageList.from_folder('../input/test/')
learn.export()
learn = load_learner(path='./', test=test)

preds, _ = learn.get_preds(ds_type=DatasetType.Test)
labelled_preds = [learn.data.classes[np.argmax(pred)] for pred in preds]
fnames = [f.name for f in learn.data.test_ds.items]
df = pd.DataFrame({'file':fnames, 'species':labelled_preds}, columns=['file', 'species'])
df.to_csv('submission.csv', index=False)