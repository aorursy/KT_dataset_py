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
print(os.listdir("../input/pixabay/"))
import fastai

from fastai.imports import *

from fastai.vision import *

from fastai.metrics import *

from fastai.gen_doc.nbdoc import *

print('fast.ai version:{}'.format(fastai.__version__))
#model = models.resnet152

model = models.densenet201

#model.load_state_dict(torch.load('../input/pretrained-pytorch-dog-and-cat-models/resnet50.pth'))

WORK_DIR = os.getcwd()

IMAGE_DIR = Path('../input/')

image_size=224

batch_size=32
labels_df = pd.read_csv('../input/labels.csv')
fns = []

ids = []

root = '../input/pixabay/'

for t in ['cats/','dogs/']:

    for l in ['0/','1/']:

        i_paths = os.listdir(f'../input/pixabay/{t}{l}')

        fns += [root + t + l + i for i in i_paths]

        ids += [p[:-4] for p in i_paths]
len(fns), len(ids)
labels_df = labels_df.set_index('id')
a = labels_df.loc[ids]
labels = a['cute'].values
data = ImageDataBunch.from_lists(path = '',fnames = fns, labels = labels,ds_tfms=get_transforms(), 

                                   test ='test',

                                   size=image_size, 

                                   bs=batch_size,

                                   num_workers=0).normalize(imagenet_stats)
data
learn = create_cnn(data, model, metrics=accuracy, model_dir=WORK_DIR)
learn.fit_one_cycle(2)
learn.recorder.plot_losses()
learn.save('stage-1')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.load('stage-1')

learn.unfreeze()

learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))
learn.recorder.plot_losses()
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9)