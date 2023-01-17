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
from fastai.vision import *
# import everything into databunch

path = Path("../input/")

tfms = get_transforms(flip_vert=True, do_flip=True)

data = ImageDataBunch.from_csv(path=path, csv_labels='train.csv', folder='train/train',

                              test='test/test',ds_tfms = tfms)

data
learn = cnn_learner(data, models.resnet50, model_dir="/tmp/model/",

                   metrics = [error_rate, accuracy])
learn.lr_find()
learn.recorder.plot()
# Let's now try to fit a first epoch.

learn.fit_one_cycle(1, 1e-2)
learn.save('fit-first-epoch')
learn.load('fit-first-epoch');
learn.unfreeze()
learn.fit_one_cycle(11, slice(1e-4, 2e-3))
learn.save('after-11-epochs');
#learn.load('after-6-epochs');
# submit a prediction

# copied from https://www.kaggle.com/interneuron/fast-fastai-with-condensenet

preds,_ = learn.get_preds(ds_type=DatasetType.Test)
test_df = pd.read_csv("../input/sample_submission.csv")

test_df.has_cactus = preds.numpy()[:, 1]
test_df.head()
to_np(p)
#submission = pd.DataFrame({'id':[i.name for i in learn.data.test_ds.items],'has_cactus': p[:,1]})
#submission.head()
#submission.to_csv('submission.csv', index=False)

test_df.to_csv('submission.csv', index=False)