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
# !head ../input/train_labels.csv
import fastai

from fastai.vision import *
batch_size = 32
tfms = get_transforms()
data = (ImageList.from_csv(csv_name='train_labels.csv', path='../input', folder='train', suffix='.jpg')

            .split_by_rand_pct()

            .label_from_df(cols='invasive')

            .transform(tfms,size=224)

            .add_test_folder('test/')

            .databunch(bs=batch_size)

           .normalize(imagenet_stats))
# data.show_batch(rows=3, figsize=(9, 9))
# data.classes, data.c
learn = cnn_learner(data, models.densenet161, metrics=accuracy, path='./')
# learn.lr_find()

# learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5,slice(1e-2))
learn.unfreeze()
# learn.lr_find()

# learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(2,slice(1e-6,1e-4))
# interp = ClassificationInterpretation.from_learner(learn)

# losses, idxs = interp.top_losses()

# interp.plot_top_losses(9, figsize=(7, 8))
# interp.plot_confusion_matrix(figsize=(3, 3))
probs, _ = learn.get_preds(ds_type=DatasetType.Test)
ilst = data.test_ds.x
fnames = [item.name.split('.')[0] for item in ilst.items]; fnames[:10]
test_df = pd.DataFrame({'name': fnames, 'invasive': probs[:,1]}) ; test_df.head()
test_df.to_csv('submission.csv', index=None)