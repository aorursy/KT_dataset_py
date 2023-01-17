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
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import torch
path = Path('../input/aptos2019-blindness-detection')

path_train = path/'train_images'

path_test = path/'test_images'

path, path_train, path_test
labels = pd.read_csv(path/'train.csv')

labels.head()
img = open_image(path_train/'000c1434d8d7.png')

img.show(figsize = (7,7))

print(img.shape)
# Distribution of the 5 diagnosis categories

labels['diagnosis'].value_counts().plot(kind = 'bar', title='Distribution of diagnosis categories')

plt.show()
# Apply data augmentation to the images

tfms = get_transforms(

    do_flip=True,

    flip_vert=True,

    max_warp=0.1,

    max_rotate=360.,

    max_zoom=1.1,

    max_lighting=0.1,

    p_lighting=0.5

)
# Applying aptos19 normalization and standard deviation stats, from a pre-trained model found on a kaggle kernel

aptos19_stats = ([0.42, 0.22, 0.075], [0.27, 0.15, 0.081])
test_labels = pd.read_csv(path/'sample_submission.csv')

test = ImageList.from_df(test_labels, path = path_test, suffix = '.png')
src = (ImageList.from_df(labels, path = path_train, suffix = '.png')

       .split_by_rand_pct(seed = 42)

       .label_from_df(cols = 'diagnosis')

       .add_test(test))
data = (

    src.transform(

        tfms,

        size = 512, 

        resize_method=ResizeMethod.SQUISH,

        padding_mode='zeros'

    )

    .databunch(bs=8)

    .normalize(aptos19_stats))
data
data.show_batch(3, figsize = (7,7))
print(data.classes)

print(len(data.train_ds))

print(len(data.valid_ds))

print(len(data.test_ds))
!mkdir -p /tmp/.cache/torch/checkpoints

!cp ../input/resnet34/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth

!cp ../input/resnet152/resnet152.pth /tmp/.cache/torch/checkpoints/resnet152-b121ed2d.pth
kappa = KappaScore()

kappa.weights = "quadratic"
learn = cnn_learner(

    data, 

    models.resnet152, 

    metrics = [accuracy, kappa], 

    model_dir = Path('../kaggle/working'),

    path = Path(".")

)
learn.fit_one_cycle(3)
learn.save('resnet152-1')
learn.load('resnet152-1');
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, slice(1e-6,1e-5))
learn.save('resnet152-2')
learn.load('resnet152-2');
# learn.export()
data = (

    src.transform(

        tfms,

        size = 1024, 

        resize_method=ResizeMethod.SQUISH,

        padding_mode='zeros'

    )

    .databunch(bs=4)

    .normalize(aptos19_stats))
learn.data = data
learn.freeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, 2e-4)
learn.save('resnet152-3')
learn.load('resnet152-3');
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6, 2e-5)
learn.save('resnet152-4')
learn.load('resnet152-4');
learn.load('resnet152-4');
preds, _ = learn.get_preds(ds_type=DatasetType.Test)
submission = pd.read_csv(path/'sample_submission.csv')

submission.head()
preds = np.array(preds.argmax(1)).astype(int).tolist()

preds[:5]
submission['diagnosis'] = preds

submission.head()
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
submission.to_csv('submission.csv', index = False)