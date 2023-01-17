%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
path = Path('../input/aerial-cactus-identification/')

import pandas as pd
train = pd.read_csv('../input/aerial-cactus-identification/train.csv')

test = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')
np.random.seed(50)

tfms = get_transforms(do_flip = True,)
data = (ImageList.from_df(train , path = path/'train' , folder = 'train')

       .split_by_rand_pct(0.01)

        .label_from_df()

        .transform(tfms, size=128)

        .databunch()).normalize(imagenet_stats)
data.show_batch(rows = 3,figsize=(7,8))
learn = cnn_learner(data , models.resnet50 , metrics = error_rate)
learn.fit_one_cycle(4)

# learn.lr_find()

# learn.recorder.plot()
test_data = ImageList.from_df(test, path=path/'test', folder='test')

data.add_test(test_data)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)

test.has_cactus = preds.numpy()[:, 0]
test.to_csv("submit.csv", index=False)
preds