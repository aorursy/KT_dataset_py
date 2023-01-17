# Importing libraries

import pandas as pd

import numpy as np

import gc

gc.enable()

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = (8.0, 5.0)



import warnings

warnings.filterwarnings("ignore")



from fastai import *

from fastai.vision import *



from utils import *
path = Path('../input/humpback-whale-identification/')

path_test = Path('../input/humpback-whale-identification/test')

path_train = Path('../input/humpback-whale-identification/train')
train_df=pd.read_csv(path/'train.csv')

val_fns = {'69823499d.jpg'}
print("Train Shape : ",train_df.shape)
print("No of Whale Classes : ",len(train_df.Id.value_counts()))
train_df.Id.value_counts().head()
(train_df.Id == 'new_whale').mean()
(train_df.Id.value_counts() == 1).mean()
fn2label = {row[1].Image: row[1].Id for row in train_df.iterrows()}

path2fn = lambda path: re.search('\w*\.jpg$', path).group(0)
gc.collect()
name = f'densenet169'



SZ = 224

BS = 64

NUM_WORKERS = 0

SEED=0
data = (

    ImageItemList

        .from_df(train_df[train_df.Id != 'new_whale'],path_train, cols=['Image'])

        .split_by_valid_func(lambda path: path2fn(path) in val_fns)

        .label_from_func(lambda path: fn2label[path2fn(path)])

        .add_test(ImageItemList.from_folder(path_test))

        .transform(get_transforms(do_flip=False, max_zoom=1, max_warp=0, max_rotate=2), size=SZ, resize_method=ResizeMethod.SQUISH)

        .databunch(bs=BS, num_workers=NUM_WORKERS, path=path)

).normalize(imagenet_stats)
data.show_batch(rows=3)
learn = create_cnn(data, models.densenet169, lin_ftrs=[2048], model_dir='../working/')

learn.clip_grad()
gc.collect()
SZ = 224 * 2

BS = 64 // 4

NUM_WORKERS = 0

SEED=0
df = pd.read_csv('../input/oversample-whale/oversampled_train_and_val.csv')
data = (

    ImageItemList

        .from_df(df, path_train, cols=['Image'])

        .split_by_valid_func(lambda path: path2fn(path) in val_fns)

        .label_from_func(lambda path: fn2label[path2fn(path)])

        .add_test(ImageItemList.from_folder(path_test))

        .transform(get_transforms(do_flip=False, max_zoom=1, max_warp=0, max_rotate=2), size=SZ, resize_method=ResizeMethod.SQUISH)

        .databunch(bs=BS, num_workers=NUM_WORKERS, path=path)

        .normalize(imagenet_stats)

)
learn = create_cnn(data, models.densenet169, lin_ftrs=[2048], model_dir='../working/')
learn.fit_one_cycle(1, slice(6.92E-06))
gc.collect()

learn.save('stage-1')
gc.collect()
preds, _ = learn.get_preds(DatasetType.Test)

preds = torch.cat((preds, torch.ones_like(preds[:, :1])), 1)

preds[:, 5004] = 0.06



classes = learn.data.classes + ['new_whale']
def top_5_preds(preds): return np.argsort(preds.numpy())[:, ::-1][:, :5]



def top_5_pred_labels(preds, classes):

    top_5 = top_5_preds(preds)

    labels = []

    for i in range(top_5.shape[0]):

        labels.append(' '.join([classes[idx] for idx in top_5[i]]))

    return labels



def create_submission(preds, data, name, classes=None):

    if not classes: classes = data.classes

    sub = pd.DataFrame({'Image': [path.name for path in data.test_ds.x.items]})

    sub['Id'] = top_5_pred_labels(preds, classes)

    sub.to_csv(f'{name}.csv', index=False)
create_submission(preds, learn.data, name, classes)