%reload_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.vision import *

from fastai.metrics import error_rate
path = Path('/kaggle/input/histopathologic-cancer-detection/')

path.ls()
path_img_train= path/'train'
path_img_test=path/'test'
path_label=path/'train_labels.csv'
df_label=pd.read_csv(path_label)
df_label.head()
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=.0, max_zoom=.1,

                      max_lighting=0.05, max_warp=0.)
doc(get_transforms)
data = ImageDataBunch.from_csv(path,folder='train',csv_labels=path_label,ds_tfms=tfms, size=90, suffix='.tif',test=path_img_test,bs=64);

stats=data.batch_stats()        

data.normalize(stats)
data.show_batch(rows=5, figsize=(12,9))
print(data.classes)

len(data.classes),data.c
data
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn
learn.fit_one_cycle(1)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12),dpi=60)
interp.most_confused(min_val=2)
learn.model_dir=Path('/kaggle/working')

learn.lr_find()
learn.recorder.plot()
learn.save('stage1')
learn.unfreeze()
learn.fit_one_cycle(1)
learn.lr_find()
learn.recorder.plot()
learn.load('stage1')
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
preds,y=learn.get_preds()
from sklearn.metrics import roc_auc_score
def auc_score(y_pred,y_true,tens=True):

    score=roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])

    if tens:

        score=tensor(score)

    else:

        score=score

    return score
pred_score=auc_score(preds,y)

pred_score
preds,y=learn.TTA()

pred_score_tta=auc_score(preds,y)

pred_score_tta
preds_test,y_test=learn.get_preds(ds_type=DatasetType.Test)
sub=pd.read_csv(f'{path}/sample_submission.csv').set_index('id')

sub.head()
clean_fname=np.vectorize(lambda fname: str(fname).split('/')[-1].split('.')[0])

fname_cleaned=clean_fname(data.test_ds.items)

fname_cleaned=fname_cleaned.astype(str)
sub.loc[fname_cleaned,'label']=to_np(preds_test[:,1])

sub.to_csv(f'submission_{pred_score}.csv')