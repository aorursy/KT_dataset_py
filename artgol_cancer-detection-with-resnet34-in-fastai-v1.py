import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from fastai.vision import *

from fastai.metrics import *

from fastai import *

from os import *



import matplotlib.pyplot as plt

import matplotlib.patches as patches

import random

from sklearn.utils import shuffle
model_path='.'

path='../input/'

train_folder=f'{path}train'

test_folder=f'{path}test'

train_lbl=f'{path}train_labels.csv'

ORG_SIZE=96



bs=64

num_workers=None # Apprently 2 cpus per kaggle node, so 4 threads I think

sz=96
df_train=pd.read_csv(train_lbl)
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=.0, max_zoom=1.1,

                      max_lighting=0.05, max_warp=0.)
data = ImageDataBunch.from_csv(path,csv_labels=train_lbl,folder='train',valid_pct=0.2, ds_tfms=tfms, size=sz, suffix='.tif',test=test_folder,bs=bs);

stats=data.batch_stats()        

data.normalize(stats)
data.show_batch(rows=5, figsize=(12,9))
from sklearn.metrics import roc_auc_score
def auc_score(y_pred,y_true,tens=True):

    score=roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])

    if tens:

        score=tensor(score)

    else:

        score=score

    return score
learn = create_cnn(data, models.resnet34, metrics=[auc_score], model_dir="/tmp/model/", ps=0.5)
learn.fit_one_cycle(1, 1e-3)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-4,1e-1)) 
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(10,10))
learn.unfreeze()
learn.fit_one_cycle(1)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, slice(5e-5,5e-3))
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(10,10))
interp.plot_confusion_matrix(figsize=(15,5))
preds,y=learn.get_preds()

pred_score=auc_score(preds,y)

pred_score
preds_test,y_test=learn.get_preds(ds_type=DatasetType.Test)
sub=pd.read_csv(f'{path}sample_submission.csv').set_index('id')

sub.head()
clean_fname=np.vectorize(lambda fname: str(fname).split('/')[-1].split('.')[0])

fname_cleaned=clean_fname(data.test_ds.items)

fname_cleaned=fname_cleaned.astype(str)
sub.loc[fname_cleaned,'label']=to_np(preds_test[:,1])

sub.to_csv(f'/kaggle/working/submission_{pred_score}.csv')