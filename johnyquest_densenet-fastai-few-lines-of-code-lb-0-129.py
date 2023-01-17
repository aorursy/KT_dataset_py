%reload_ext autoreload
%autoreload 2
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastai import *
from fastai.vision import *
from fastai.vision.image import *
path3 = Config.data_path()/'protien_atlas'
path3
path4 = Config.data_path()/'protien_atlas/test'
path4
df = pd.read_csv(path3/'train.csv')
df.head()
tfms = get_transforms(flip_vert=True, max_lighting=0.2, max_zoom=1.05, max_warp=0.)
#Planning to try the following tfms, not sure it will make much changes 
#tfms2 =get_transforms(flip_vert=True, max_rotate=30., max_zoom=1, max_lighting=0.05, max_warp=0.)
np.random.seed(42)
src = (ImageItemList.from_csv('/home/jupyter/.fastai/data', 'train.csv', folder='protien_atlas', suffix='.png')
       .random_split_by_pct(0.2)
       .label_from_df(sep=' ',  classes=[str(i) for i in range(28)]))
test_ids = list(sorted({fname.split('_')[0] for fname in os.listdir(path4)}))
test_fnames = [path4/test_id for test_id in test_ids]
test_fnames[:5]
src.add_test(test_fnames, label='0');
data = (src.transform(tfms, size=128)
        .databunch(num_workers=0).normalize(imagenet_stats))
print('Train size:', len(data.train_ds))
print('Valid size:', len(data.valid_ds))
print('Test size:', len(data.test_ds))
data.show_batch(rows=3, figsize=(12,9))
import fastai.vision
from torchvision.models import densenet201 
def _densenet201_split(m:nn.Module): return (m[0][0][7], m[1]) 
_densenet201_meta  = {'cut':-1, 'split': _densenet201_split} 
fastai.vision.learner.model_meta = { densenet201:{**_densenet201_meta} }
arch = densenet201
f1_score = partial(fbeta, thresh=0.2, beta=2)
learn = create_cnn(data, arch, metrics= [f1_score])
learn.fit(5)
learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-5,3e-4), pct_start=0.05)
preds,_ = learn.get_preds(DatasetType.Test)
pred_labels = [' '.join(list([str(i) for i in np.nonzero(row>0.2)[0]])) for row in np.array(preds)]
df = pd.DataFrame({'Id':test_ids,'Predicted':pred_labels})
df.to_csv('submission2.csv', header=True, index=False)
pred_test_tta,_=learn.TTA(ds_type=DatasetType.Test)
pred_labels2 = [' '.join(list([str(i) for i in np.nonzero(row>0.2)[0]])) for row in np.array(pred_test_tta)]
df2 = pd.DataFrame({'Id':test_ids,'Predicted':pred_labels2})
df2.to_csv('submission5.csv', header=True, index=False)
pred,y=learn.get_preds()
sample_csv = f'sample_submission.csv'
th_t = np.array([0.565,0.39,0.55,0.345,0.33,0.39,0.33,0.45,0.38,0.39,
               0.34,0.42,0.31,0.38,0.49,0.50,0.38,0.43,0.46,0.40,
               0.39,0.505,0.37,0.47,0.41,0.545,0.32,0.1])
from sklearn.metrics import f1_score as f1_sc
def eval_pred(pred,y):
    ths=np.arange(0.001,1,0.01)
    preds_s=F.sigmoid(pred)
    th_val=ths[np.argmax([f1_sc(y,preds_s>th,average='macro') for th in ths])]
    print('F1 macro: ',f1_sc(to_np(y),to_np(preds_s)>th_t,average='macro'))
    print(f'F1 macro (th = {th_val}): ',f1_sc(to_np(y),to_np(preds_s)>th_val,average='macro'))
    plt.plot(f1_sc(to_np(y),to_np(preds_s)>th_t,average=None),label='opt')
    plt.plot(f1_sc(to_np(y),to_np(preds_s)>th_val,average=None),label='valid')
    plt.legend()
ths=np.arange(0.001,1,0.01)
preds_s=F.sigmoid(pred)
th_val=ths[np.argmax([f1_sc(y,preds_s>th,average='macro') for th in ths])]
eval_pred(pred,y)
def save_pred(pred, th=0.5, fname='protein_classification.csv'):
    pred_list = []
    for line in pred:
        s = ' '.join(list([str(i) for i in np.nonzero(line>th)[0]]))
        pred_list.append(s)
        
    sample_df = pd.read_csv(sample_csv)
    sample_list = list(sample_df.Id)
    #fnames_=[fname.split('/')[-1] for fname in learn.data.test_ds.fnames]
    pred_dic = dict((key, value) for (key, value) 
                in zip(test_ids,pred_list))
    pred_list_cor = [pred_dic[id] for id in test_ids]
    df = pd.DataFrame({'Id':sample_list,'Predicted':pred_list_cor})
    df.to_csv(fname, header=True, index=False)
save_pred(to_np(F.sigmoid(preds)), th=th_val, fname=f'protein_classification_{np.around(th_val,decimals=2)}.csv')

save_pred(to_np(F.sigmoid(preds)), th=th_t, fname='protein_classification_customth.csv')
save_pred(to_np(F.sigmoid(pred_test_tta)), th=th_val, fname=f'protein_classification_{np.around(th_val,decimals=2)}_tta.csv')

save_pred(to_np(F.sigmoid(pred_test_tta)), th=th_t, fname='protein_classification_customth_tta.csv')
