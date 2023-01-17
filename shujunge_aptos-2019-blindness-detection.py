%matplotlib inline

import os

import pandas as pd

from glob import glob

import numpy as np

from fastai import *

from fastai.vision import *

import torch 

import matplotlib.pyplot as plt
path = Path('../input/aptos2019-blindness-detection')

path_train = path/'train_images'

path_test = path/'test_images'

path, path_train, path_test
labels = pd.read_csv(path/'train.csv')

labels.head()
img = open_image(path_train/'000c1434d8d7.png')

img.show(figsize = (7,7))

print(img.shape)
labels['diagnosis'].value_counts().plot(kind = 'bar', title='Distribution of diagnosis categories')

plt.show()
tfms = get_transforms(

    do_flip=True,

    flip_vert=True,

    max_warp=0.1,

    max_rotate=66.,

    max_zoom=1.1,

    max_lighting=0.1,

    p_lighting=0.5

)

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

        size = 128, 

        resize_method=ResizeMethod.SQUISH,

        padding_mode='zeros'

    )

    .databunch(bs=32)

    .normalize(aptos19_stats))
data.show_batch(3, figsize = (7,7))
print(data.classes)

print(len(data.train_ds))

print(len(data.valid_ds))

print(len(data.test_ds))
!mkdir models

!mkdir -p /tmp/.cache/torch/checkpoints

!cp ../input/resnet34/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth
kappa = KappaScore()

kappa.weights = "quadratic"
learn = cnn_learner(

    data, 

    models.resnet34, 

    metrics = [accuracy, kappa], 

    model_dir = Path('../kaggle/working'),

    path = Path(".")

)
learn.fit_one_cycle(15)

learn.save('resnet34')
learn.load('resnet34')

learn.unfreeze()

learn.lr_find()

learn.recorder.plot()

learn.fit_one_cycle(20, slice(1e-6,5e-4))

learn.save('resnet34-1')
# learn.load('resnet152-2')

# data = (

#     src.transform(

#         tfms,

#         size = 1024, 

#         resize_method=ResizeMethod.SQUISH,

#         padding_mode='zeros'

#     )

#     .databunch(bs=4)

#     .normalize(aptos19_stats))

# learn.data = data

# learn.freeze()

# learn.lr_find()

# learn.recorder.plot()

# learn.fit_one_cycle(4, 2e-4)

# learn.save('resnet152-3')
# learn.load('resnet152-3')

# learn.unfreeze()

# learn.lr_find()

# learn.recorder.plot()

# learn.fit_one_cycle(6, 2e-5)

# learn.save('resnet152-4')

learn.load('resnet34-1')

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