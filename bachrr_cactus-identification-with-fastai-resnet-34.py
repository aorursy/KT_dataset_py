# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import fastai

from fastai.vision import *



# Any results you write to the current directory are saved as output.
print(fastai.__version__)
PATH = Path('.')

!ls {PATH}
df = pd.read_csv(PATH/'../input/train.csv'); df.head()
df['has_cactus'].hist()
bs = 128
tfms = get_transforms(do_flip=True, flip_vert=True)
data = (ImageList.from_csv(csv_name='train.csv', path=PATH/'../input', folder='train/train')

            .split_by_rand_pct()

            .label_from_df(cols='has_cactus')

            .transform(tfms)

            .add_test_folder('test/test')

            .databunch(bs=bs)

           .normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(7, 7))
data.classes, data.c
learn = cnn_learner(data, models.resnet34, metrics=accuracy, path=PATH)
learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()

interp.plot_top_losses(9, figsize=(7, 8))
interp.plot_confusion_matrix(figsize=(3, 3))
learn.unfreeze()
learn.fit_one_cycle(1)
learn.recorder.plot()
probs, preds = learn.get_preds(ds_type=DatasetType.Test)
preds.shape, probs.shape
!ls {PATH}/../input/test/test | wc -l
ilst = data.test_ds.x
fnames = [item.name for item in ilst.items]; fnames[:10]
test_df = pd.DataFrame({'id': fnames, 'has_cactus': probs[:, 1]}); test_df
test_df.to_csv('submission.csv', index=None)
!head submission.csv