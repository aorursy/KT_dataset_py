# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import cv2

import os

from fastai.vision import *

from fastai.metrics import KappaScore

from pathlib import Path
base_path = Path("/kaggle/input/aptos2019-blindness-detection")

out_path = Path("/kaggle/working")
train = pd.read_csv(base_path/"train.csv")

train.head()
test = pd.read_csv(base_path/"test.csv")

test.head()

train['diagnosis'].value_counts()
kappa = KappaScore(weights="quadratic")
data = (ImageList.from_df(train, base_path, folder='train_images', suffix=".png")

                   .split_by_rand_pct()

                   .label_from_df()

                   .transform(get_transforms(), size=256)

                   .databunch(bs=64)

     ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using device:", device)
learn = cnn_learner(data, models.resnet50, metrics=[kappa], pretrained=True, model_dir=out_path)

learn.model.cuda();

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=3e-3)

learn.save(out_path/"stage-1")

learn.recorder.plot_losses()
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=3e-4)

learn.save(out_path/"stage-2")

learn.recorder.plot_losses()
submission = pd.read_csv(base_path/"sample_submission.csv")

learn.data.add_test(ImageList.from_df(submission,Path(base_path),folder='test_images',suffix='.png'))



preds = []

from tqdm import tqdm_notebook as tqdm

for i in tqdm(learn.data.test_ds):

    preds.append(int(learn.predict(i[0])[1]))

print(len(preds))



submission["diagnosis"] = np.asarray(preds).reshape(-1, 1)

submission.to_csv('submission.csv', index=False)