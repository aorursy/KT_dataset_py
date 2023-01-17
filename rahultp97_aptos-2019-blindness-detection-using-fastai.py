# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from fastai.metrics import accuracy, KappaScore

from fastai.vision import *

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
!ls ../input/resnet34/
Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)

!cp ../input/resnet34/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34.pth
!mv  /tmp/.cache/torch/checkpoints/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

train_df.head()
add_extension = lambda x: str(x) + '.png'

add_dir = lambda x: os.path.join('train_images', x)
train_df['diagnosis']=train_df['diagnosis'].apply(int)

train_df['id_code'] = train_df['id_code'].apply(add_extension)

train_df['id_code'] = train_df['id_code'].apply(add_dir)
train_df.head()
data_dir = Path('../input/aptos2019-blindness-detection/')
data = ImageDataBunch.from_df(data_dir, train_df, valid_pct=0.2, 

                              label_col='diagnosis', ds_tfms=get_transforms(), 

                              size=224).normalize(imagenet_stats)
data.classes
kappa = KappaScore()

kappa.weights = "quadratic"
learn = cnn_learner(data, models.resnet34, metrics=[accuracy, kappa], model_dir = Path('/kaggle/working/'))
learn.fit_one_cycle(5)

learn.recorder.plot_losses()
import gc

gc.collect()
# learn.unfreeze()
# learn.lr_find()
# learn.recorder.plot()
# learn.fit_one_cycle(8, max_lr=slice(8e-6, 2e-5))

# learn.recorder.plot_losses()
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

test_df.head()
test_df['diagnosis'] = pd.Series(np.zeros(len(test_df['id_code'])), index=test_df.index)
test_df.head()
learn.data.add_test(ImageList.from_df(test_df,'../input/aptos2019-blindness-detection/',folder='test_images',suffix='.png'))
preds,y = learn.get_preds(DatasetType.Test)
test_df.diagnosis = preds.argmax(1)

test_df.head()
test_df.to_csv('submission.csv',index=False)