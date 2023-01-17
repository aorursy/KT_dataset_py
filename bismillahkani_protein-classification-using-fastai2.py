!pip install fastai2 -q
%load_ext autoreload
%autoreload 2

import os
import pandas as pd
import numpy as np

from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *
from fastai2.vision.widgets import *
from fastai2.vision import *
from fastai2.callback.cutmix import *
DATA_DIR = '../input/jovian-pytorch-z2g/Human protein atlas'

TRAIN_DIR = DATA_DIR + '/train'                           
TEST_DIR = DATA_DIR + '/test'                             

TRAIN_CSV = DATA_DIR + '/train.csv'                       
TEST_CSV = '../input/jovian-pytorch-z2g/submission.csv' 
train_df = pd.read_csv(TRAIN_CSV)
train_df['imgPath'] = train_df.apply(lambda x : os.path.join(TRAIN_DIR,str(x['Image'])+'.png'),axis=1)
train_df.head()
split_df = pd.get_dummies(train_df.Label.str.split(" ").explode())
split_df = split_df.groupby(split_df.index).sum()
split_df.head()
X, y = split_df.index.values, split_df.values
from skmultilearn.model_selection import IterativeStratification

nfolds = 5

k_fold = IterativeStratification(n_splits=nfolds, order=1)

splits = list(k_fold.split(X, y))

fold_splits = np.zeros(train_df.shape[0]).astype(np.int)

for i in range(nfolds):
    fold_splits[splits[i][1]] = i
train_df['Split'] = fold_splits
def get_fold(fold):
    train_df['is_valid'] = False
    train_df.loc[train_df.Split == fold, 'is_valid'] = True
    return train_df
train_df.head()
train_df = get_fold(0)
train_df = train_df.drop(['Split'],axis=1)
train_df.head()
test_df = pd.read_csv(TEST_CSV)
test_df['imgPath'] = test_df.apply(lambda x : os.path.join(TEST_DIR,str(x['Image'])+'.png'),axis=1)
test_df.head()
labels = {
    0: 'Mitochondria',
    1: 'Nuclear bodies',
    2: 'Nucleoli',
    3: 'Golgi apparatus',
    4: 'Nucleoplasm',
    5: 'Nucleoli fibrillar center',
    6: 'Cytosol',
    7: 'Plasma membrane',
    8: 'Centrosome',
    9: 'Nuclear speckles'
}
def encode_label(label):
    target = torch.zeros(10)
    for l in str(label).split(' '):
        target[int(l)] = 1.
    return target

def decode_target(target, text_labels=False, threshold=0.5):
    result = []
    for i, x in enumerate(target):
        if (x >= threshold):
            if text_labels:
                result.append(labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return ' '.join(result)
aug_tfms = aug_transforms(mult=1.0, 
               do_flip=True, 
               flip_vert=False, 
               max_rotate=10.0, 
               max_zoom=1.1, 
               max_lighting=0.5, 
               max_warp=0.2, 
               p_affine=0.75, 
               p_lighting=0.75, 
               xtra_tfms=RandomErasing(p=1., max_count=6), 
               size=224, 
               mode='bilinear', 
               pad_mode='reflection', 
               align_corners=True, 
               batch=False, 
               min_scale=0.75)
def get_x(r): return r['imgPath']
def get_y(r): return r['Label'].split(' ')
def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train,valid
dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter, 
                   get_x = get_x, get_y = get_y,
                   item_tfms=Resize(460),
                   batch_tfms=aug_tfms)
dls = dblock.dataloaders(train_df)
dls.train.show_batch(max_n=9)
def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Compute accuracy when `inp` and `targ` are the same size."
    if sigmoid: inp = inp.sigmoid()
    return ((inp>thresh)==targ.bool()).float().mean()
def F_score(output, label, threshold=0.2, beta=1):
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0)
cutmix = CutMix(1.0)
learn = cnn_learner(dls, resnet18, 
                    metrics=[partial(accuracy_multi, thresh=0.2),partial(F_score, threshold=0.2)],
                    cbs=cutmix,
                   callback_fns=[partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.01, patience=3)])
learn._do_begin_fit(1)
learn.epoch,learn.training = 0,True
learn.dl = dls.train
b = dls.one_batch()
learn._split(b)
learn('begin_batch')
_,axs = plt.subplots(3,3, figsize=(9,9))
dls.show_batch(b=(cutmix.x,cutmix.y), ctxs=axs.flatten())
learn.fine_tune(3, base_lr=3e-3, freeze_epochs=2)
mixup = MixUp(0.4) 
learn = cnn_learner(dls, resnet18, 
                    metrics=[partial(accuracy_multi, thresh=0.2),partial(F_score, threshold=0.2)],
                    cbs=mixup,
                   callback_fns=[partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.01, patience=3)])
learn._do_begin_fit(1)
learn.epoch,learn.training = 0,True
learn.dl = dls.train
b = dls.one_batch()
learn._split(b)
learn('begin_batch')
_,axs = plt.subplots(3,3, figsize=(9,9))
dls.show_batch(b=(mixup.x,mixup.y), ctxs=axs.flatten())
learn.fine_tune(3, base_lr=3e-3, freeze_epochs=2)
learn = cnn_learner(dls, resnet18, 
                    metrics=[partial(accuracy_multi, thresh=0.2),partial(F_score, threshold=0.2)],
                   callback_fns=[partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.01, patience=3)])
learn.fine_tune(3, base_lr=3e-3, freeze_epochs=4)
preds,targs = learn.get_preds()
xs = torch.linspace(0.05,0.95,29)
accs = [accuracy_multi(preds, targs, thresh=i, sigmoid=False) for i in xs]
fscores = [F_score(preds, targs, threshold=i, beta=1) for i in xs]
plt.plot(xs,accs);
plt.plot(xs,fscores);
learn = cnn_learner(dls, resnet18, 
                    metrics=[partial(accuracy_multi, thresh=0.2),partial(F_score, threshold=0.2)],
                   callback_fns=[partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.01, patience=3)])
learn.fine_tune(20, base_lr=3e-3, freeze_epochs=4)
learn.show_results()
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(5, nrows=1)
learn.export('export.pkl')
dl = learn.dls.test_dl(test_df)
preds,targs = learn.tta(dl=dl)
predictions = [decode_target(x, threshold=0.5) for x in preds]
submission_df = pd.read_csv(TEST_CSV)
submission_df.Label = predictions
submission_df.head()
sub_fname = 'submission_fastai_v6_1.csv'
submission_df.to_csv(sub_fname, index=False)
predictions = [decode_target(x, threshold=0.2) for x in preds]
submission_df = pd.read_csv(TEST_CSV)
submission_df.Label = predictions
submission_df.head()
sub_fname = 'submission_fastai_v6_2.csv'
submission_df.to_csv(sub_fname, index=False)
!pip install jovian --upgrade --quiet
import jovian
project_name='protein-advanced'
jovian.commit(project=project_name, environment=None)
