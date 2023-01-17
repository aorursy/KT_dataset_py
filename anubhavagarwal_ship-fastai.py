# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/dataset_aug/dataset_aug"))#

#print(os.listdir("/tmp/models/"))



# Any results you write to the current directory are saved as output.
from fastai.vision import *

from fastai.metrics import error_rate, accuracy

import numpy as np

bs = 64

tfms = get_transforms(max_rotate=20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4,

                      p_affine=1., p_lighting=1.)



path = "../input/dataset_aug/dataset_aug"



data = ImageDataBunch.from_folder(path,bs=bs//2,

                                  ds_tfms=tfms, size=200, num_workers=4,train='train', valid='validation', test='test').normalize(imagenet_stats)



data.classes

dir(models)
data
doc(data.show_batch)

data.show_batch(rows=3, figsize=(7, 8))

#https://github.com/fastai/imagenet-fast/blob/master/cifar10/models/inceptionresnetv2.py

data.classes, data.c, len(data.train_ds), len(data.valid_ds)

learn = create_cnn(data, models.resnet152, metrics=accuracy,model_dir='/tmp/models/', ps=0.5)

#learn = create_cnn(data, models.densenet201, metrics=accuracy,model_dir='/tmp/models/', ps=0.5)

#learn = create_cnn(data, models.xresnet152, metrics=accuracy,model_dir='/tmp/models/', ps=0.5, pretrained=False)

defaults.device = torch.device('cuda') # makes sure the gpu is used

learn.lr_find()

learn.recorder.plot(suggestion=True)
#learn.fit_one_cycle(4)

#learn.unfreeze()

learn.fit_one_cycle(10, max_lr=slice(3.63E-03, 1e-01))
#learn.save("/tmp/models/res_save_2")

#learn = learn.load("/tmp/models/res_save_2")
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)

learn.fit_one_cycle(10, max_lr=slice(6.31E-07, 1e-035))

#learn.save("/tmp/models/res_save_2")

learn = learn.load("/tmp/models/res_save_2")
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, max_lr=slice(6.92E-06, 1e-05))
#learn.save("/tmp/models/res_save_3")

learn = learn.load("/tmp/models/res_save_3")
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, max_lr=slice(6.31E-07, 1e-06))
preds, _ = learn.TTA(ds_type=DatasetType.Test)
#preds, _ = learn.get_preds(ds_type=DatasetType.Test)

import numpy as np

pred = np.argmax(preds, axis=1) 

test = Path("../input/dataset_aug/dataset_aug/test")

test_imgs = [i for i in test.iterdir()]

import pandas as pd

ids = np.array([f.name for f in (test_imgs)])

ids.shape

sub = pd.DataFrame(np.stack([ids, pred + 1], axis=1), columns=['image','category'])

sub.to_csv('submission22.csv', index=False)