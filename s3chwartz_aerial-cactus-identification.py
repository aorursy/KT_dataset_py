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



from pathlib import Path

import fastai

from fastai.vision import *

from fastai.metrics import error_rate

PATH = Path("../input")



work_dir = Path("/kaggle/working/")

train = "train/train"

test = PATH/"test/test"

test_names = [f for f in test.iterdir()]



df_train = pd.read_csv(PATH/"train.csv")



submission = pd.read_csv(PATH/"sample_submission.csv")

#train.head()
data = (ImageList.from_df(df_train, path=PATH/train, cols=0).split_by_rand_pct(0.2, seed=47)

        .label_from_df(cols=1)

        .add_test(test_names)

        .databunch(bs=128))

data.normalize(imagenet_stats)

#.add_test(test_names)

data
learn = cnn_learner(data, models.resnet50,

                    metrics = accuracy,

                   model_dir="/tmp/model/")
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(50, max_lr=slice(1e-6,1e-1))
learn.save("fit_resnet50_v1")
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

interp.plot_top_losses(9, figsize=(15,11))
p, t = learn.get_preds(ds_type = DatasetType.Test)
ids = np.array([f.name for f in (test_names)])

ids.shape
pmax = np.argmax(p, 1)

pmax.shape
my_submission = pd.DataFrame({"id": ids,

                             "has_cactus" : pmax})

my_submission.to_csv("submission.csv", index = False)