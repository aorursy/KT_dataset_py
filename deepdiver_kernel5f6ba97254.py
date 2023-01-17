# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd
from fastai import *

from fastai.vision import *
path = Path('../input/')

train_df = pd.read_csv(path/'train.csv')

test_df = pd.read_csv(path/'test.csv')
train, test = [ImageList.from_df(df, path=path, cols='img_file', folder=folder) 

               for df, folder in zip([train_df, test_df], ['train', 'test'])]



tfms = get_transforms(do_flip=True, flip_vert=False, max_rotate=0.20, max_zoom=2, max_lighting=0.1)



data = (train.split_by_rand_pct(0.1, seed=42)

        .label_from_df(cols='class')

        .add_test(test)

        .transform(tfms, size=224)

        .databunch(path=Path('.'), bs=64).normalize(imagenet_stats))
data.show_batch(rows=4)
print(data.classes)

len(data.classes), data.c
learn = cnn_learner(data, models.densenet201, metrics=[accuracy, error_rate])
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-06, 1e-02))
learn.save('stage-1')
learn.load('stage-1')

learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(10, max_lr=slice(1e-07, 1e-04))
learn.save('stage-2')
learn.unfreeze()

learn.lr_find(start_lr=1e-10)
learn.recorder.plot()
learn.fit(epochs=10, lr=1e-4)
learn.save('stage-3')
learn.lr_find()
learn.recorder.plot()
learn.fit(epochs=5, lr=1e-5)
learn.save('stage-4')
test_preds = learn.TTA(ds_type=DatasetType.Test)

test_df['class'] = np.argmax(test_preds[0], axis=1) + 1

test_df.head()
test_df.to_csv('submission.csv', index=False) 
learn.load('stage-4')

learn.fit(epochs=10, lr=1e-7)