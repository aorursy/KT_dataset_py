import numpy as np

import pandas as pd

from fastai import *

from fastai.vision import *

from torchvision.models import *

import os
df = pd.read_csv('../input/train_labels.csv')

df.head()
df_unique = pd.unique(df['label'])

df_unique
path = Path('../input')

path.ls()
SZ = 96

BS = 64

NUM_WORKERS = 4
data = (

    ImageItemList

        .from_csv(path, 'train_labels.csv', folder='train', suffix='.tif')

        .random_split_by_pct()

        .label_from_df()

        .add_test(ImageItemList.from_folder('../input/test'))

        .transform(get_transforms(do_flip=True, flip_vert=True, max_rotate=.0, max_zoom=.1,max_lighting=0.05, max_warp=0.), size=SZ, resize_method=ResizeMethod.SQUISH)

        .databunch(bs=BS, num_workers=NUM_WORKERS, path = '.')

        .normalize(imagenet_stats)

)
data.show_batch(3)
from sklearn.metrics import roc_auc_score

def auc_score(y_pred,y_true,tens=True):

    score=roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])

    if tens:

        score=tensor(score)

    else:

        score=score

    return score
learn = create_cnn(data, densenet161, metrics=[auc_score], ps=0.5)
# learn.lr_find()

# learn.recorder.plot()
learn.fit_one_cycle(1, 1e-3)
learn.unfreeze()

# learn.lr_find()

# learn.recorder.plot()
learn.fit_one_cycle(4, slice(1e-6, 1e-4))
SZ = 96*2

BS = 64//2

NUM_WORKERS = 0
data = (

    ImageItemList

        .from_csv(path, 'train_labels.csv', folder='train', suffix='.tif')

        .random_split_by_pct()

        .label_from_df()

        .add_test(ImageItemList.from_folder('../input/test'))

        .transform(get_transforms(do_flip=True, flip_vert=True, max_rotate=.0, max_zoom=.1,max_lighting=0.05, max_warp=0.), size=SZ, resize_method=ResizeMethod.SQUISH)

        .databunch(bs=BS, num_workers=NUM_WORKERS, path = '.')

        .normalize(imagenet_stats)

)
learn.data = data
# learn.lr_find()

# learn.recorder.plot()
learn.freeze()
learn.fit_one_cycle(1, 1e-3/2)
learn.unfreeze()
learn.fit_one_cycle(1, slice(1e-6/2, 1e-4/2))
preds = learn.get_preds(ds_type=DatasetType.Test)
df_sub = pd.read_csv('../input/sample_submission.csv').set_index('id')

df_sub.head()
clean_fname = np.vectorize(lambda fname: str(fname).split('/')[-1].split('.')[0])

fname_cleaned = clean_fname(data.test_ds.items)

fname_cleaned = fname_cleaned.astype(str)
df_sub.loc[fname_cleaned,'label'] = preds[0][:,1].tolist()
df_sub.head()
df_sub.to_csv('sub.csv')