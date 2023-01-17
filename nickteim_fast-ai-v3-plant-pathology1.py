%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *



import os

import pandas as pd

import sys



from collections import Counter

from pathlib import Path

import numpy
path = Path('/kaggle/input/plant-pathology-2020-fgvc7')

path.ls()
path2 = Path('/kaggle/input/plant-pathology-2020-fgvc7/images')

# path2.ls()
df = pd.read_csv(path/'train.csv')

df.head()
test_df = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')
#transform the data to get a bigger training set

tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
# labels for our classes

LABEL_COLS = ['healthy', 'multiple_diseases', 'rust', 'scab']
#dataloading of testset

test = (ImageList.from_df(test_df,path,

                          folder='images',

                          suffix='.jpg',

                          cols='image_id'))
test
#making training set with API Block

np.random.seed(42)

src=(ImageList.from_csv(path,'train.csv',folder='images',suffix='.jpg')

    .split_by_rand_pct(0.2)

    .label_from_df(cols=LABEL_COLS,label_cls = MultiCategoryList))
data = (src.transform(tfms, size=253).add_test(test)

        .databunch(num_workers=0).normalize(imagenet_stats))
data.classes
data.show_batch(rows=3, figsize=(12,9))
arch = models.resnet50
acc_02 = partial(accuracy_thresh, thresh=0.2)

f_score = partial(fbeta, thresh=0.2)

learn = create_cnn(data, arch, metrics=[acc_02, f_score], model_dir='/kaggle/working')
# learn.lr_find()

# learn.recorder.plot()
lr=1e-03

learn.fit_one_cycle(5,slice(lr))
learn.save('stage-1-rn50')
learn.unfreeze()
# learn.lr_find()

# learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-5, 1e-3))
learn.save('stage-2-rn50')
tfms = get_transforms()
data = (src.transform(tfms, size=399)

        .databunch().normalize(imagenet_stats))



learn.data = data

data.train_ds[0][0].shape
learn.freeze()
# learn.lr_find()

# learn.recorder.plot()
lr=1e-3

learn.fit_one_cycle(5, slice(lr))
learn.unfreeze()
# learn.lr_find()

# learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-6, 1e-4))
learn.save('stage-2-256-rn50')
#plot losses to see how good the model trained

learn.recorder.plot_losses()
#Make predictions on test set

preds = learn.get_preds(DatasetType.Test)
test = pd.read_csv(path/'test.csv')

test_id = test['image_id'].values
submission = pd.DataFrame({'image_id': test_id})

submission = pd.concat([submission, pd.DataFrame(preds[0].numpy(), columns = LABEL_COLS)], axis=1)

submission.to_csv('submission_plant12.csv', index=False)

submission.head(10)

print('Model ready for submission!')