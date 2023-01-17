!pip install git+https://github.com/fastai/fastcore > /dev/null
!pip install git+https://github.com/fastai/fastai2 > /dev/null
!pip install iterative-stratification > /dev/null
!pip install efficientnet-pytorch
%load_ext autoreload
%autoreload 2

import os
import pandas as pd
import sys
from collections import Counter
from pathlib import Path

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm.notebook import tqdm
from torchvision.models import densenet121
from sklearn.metrics import roc_auc_score
from torch.utils.data.sampler import WeightedRandomSampler

from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
from efficientnet_pytorch import EfficientNet
#from fastai2.vision.all import *
DATA_PATH = Path('../input/plant-pathology-2020-fgvc7')
IMG_PATH = DATA_PATH / 'images'
LABEL_COLS = ['healthy', 'multiple_diseases', 'rust', 'scab']

IMG_SIZE = 512
SEED = 420
N_FOLDS = 5
BS = 8#16
N_FOLDS = 5

ARCH = densenet121
seed_everything(SEED)
train_df = pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')
test_df = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')
train_df.head()
(len(train_df), len(test_df))
_, axes = plt.subplots(ncols=4, nrows=1, constrained_layout=True, figsize=(10, 3))
for ax, column in zip(axes, LABEL_COLS):
    train_df[column].value_counts().plot.bar(title=column, ax=ax)
plt.show()
train_df.iloc[:,1:-1].sum(axis=1).value_counts()
train_df[['healthy', 'multiple_diseases', 'rust', 'scab']].sum(axis=1).unique()
test_df.head()
# hs, ws = [], []
# for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
#     img = Image.open(IMG_PATH/(row.image_id+'.jpg'))
#     h, w = img.size
#     hs.append(h)
#     ws.append(w)
#set(hs), set(ws)
# _, axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True, figsize=(10, 3))
# for ax, column, vals in zip(axes, ['heights', 'widths'], [hs, ws]):
#     ax.hist(vals, bins=100)
#     ax.set_title(f'{column} hist')

# plt.show()
# Counter(hs), Counter(ws)
# red_values = []; green_values = []; blue_values = []; all_channels = []
# for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
#     img = np.array(Image.open(IMG_PATH/(row.image_id+'.jpg')))
#     red_values.append(np.mean(img[:, :, 0]))
#     green_values.append(np.mean(img[:, :, 1]))
#     blue_values.append(np.mean(img[:, :, 2]))
#     all_channels.append(np.mean(img))
# _, axes = plt.subplots(ncols=4, nrows=1, constrained_layout=True, figsize=(16, 3), sharey=True)
# for ax, column, vals, c in zip(
#     axes,
#     ['red', 'green', 'blue', 'all colours'],
#     [red_values, green_values, blue_values, all_channels],
#     'rgbk'
# ):
#     ax.hist(vals, bins=100, color=c)
#     ax.set_title(f'{column} hist')

# plt.show()
train_df['fold'] = -1

strat_kfold = MultilabelStratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
for i, (_, test_index) in enumerate(strat_kfold.split(train_df.image_id.values, train_df.iloc[:,1:].values)):
    train_df.iloc[test_index, -1] = i
    
train_df['fold'] = train_df['fold'].astype('int')
train_df.fold.value_counts().plot.bar()
train_df.to_csv('train_with_strat_folds.csv', index=False)
def get_label(row):
    for k, v in row[LABEL_COLS].items():
        if v == 1:
            return k
train_df['label'] = train_df.apply(get_label, axis=1)
def get_data(fold):
    train_df_no_val = train_df.query(f'fold != {fold}')
    train_df_just_val = train_df.query(f'fold == {fold}')

    train_df_bal = pd.concat(
        [train_df_no_val.query('label != "multiple_diseases"'), train_df_just_val] +
        [train_df_no_val.query('label == "multiple_diseases"')] * 4 # back to 4 as this was hs
    ).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    datablock = DataBlock(
        blocks=(ImageBlock, CategoryBlock(vocab=LABEL_COLS)),
        getters=[
            ColReader('image_id', pref=IMG_PATH, suff='.jpg'),
            ColReader('label')
        ],
        splitter=IndexSplitter(train_df_bal.loc[train_df_bal.fold==fold].index),
        item_tfms=Resize(IMG_SIZE),
        batch_tfms=aug_transforms(size=IMG_SIZE, max_rotate=30., min_scale=0.75, flip_vert=True, do_flip=True)
    )
    return datablock.dataloaders(source=train_df_bal, bs=BS)
def get_data_larger(fold):
    train_df_no_val = train_df.query(f'fold != {fold}')
    train_df_just_val = train_df.query(f'fold == {fold}')

    train_df_bal = pd.concat(
        [train_df_no_val.query('label != "multiple_diseases"'), train_df_just_val] +
        [train_df_no_val.query('label == "multiple_diseases"')] * 4 # back to 4 as this was hs
    ).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    datablock = DataBlock(
        blocks=(ImageBlock, CategoryBlock(vocab=LABEL_COLS)),
        getters=[
            ColReader('image_id', pref=IMG_PATH, suff='.jpg'),
            ColReader('label')
        ],
        splitter=IndexSplitter(train_df_bal.loc[train_df_bal.fold==fold].index),
        item_tfms=Resize(IMG_SIZE*2),
        batch_tfms=aug_transforms(size=IMG_SIZE*2, max_rotate=30., min_scale=0.75, flip_vert=True, do_flip=True)
    )
    return datablock.dataloaders(source=train_df_bal, bs=BS/2)
dls = get_data(fold=0)
dls.c
#dls.show_batch()
def comp_metric(preds, targs, labels=range(len(LABEL_COLS))):
    # One-hot encode targets
    targs = np.eye(4)[targs]
    return np.mean([roc_auc_score(targs[:,i], preds[:,i]) for i in labels])

def healthy_roc_auc(*args):
    return comp_metric(*args, labels=[0])

def multiple_diseases_roc_auc(*args):
    return comp_metric(*args, labels=[1])

def rust_roc_auc(*args):
    return comp_metric(*args, labels=[2])

def scab_roc_auc(*args):
    return comp_metric(*args, labels=[3])
def get_learner(fold_num, lr=1e-3):
    opt_func = partial(Adam, lr=lr, wd=0.01, eps=1e-8)

    data = get_data(fold_num)
    
    model = EfficientNet.from_pretrained("efficientnet-b7", advprop=True)
    #model = EfficientNet.from_name("efficientnet-b7")
    #model = EfficientNet.from_pretrained("efficientnet-b8", advprop=True) # weights run to NaN
    #model = EfficientNet.from_name('efficientnet-b4') 
    #model._fc = nn.Linear(1280, data.c)# the last layer... # works for b0,b1
    #model._fc = nn.Linear(1536, data.c)# the last layer... B3
    #model._fc = nn.Linear(1792, data.c)# the last layer... B4
    #model._fc = nn.Linear(2048, data.c)# the last layer... B5
    #model._fc = nn.Linear(2304, data.c)# the last layer... B6
    model._fc = nn.Linear(2560, data.c)# the last layer... B7
    #model._fc = nn.Linear(2816, data.c)# the last layer... B8

    learn = Learner(
        dls, model, opt_func=opt_func,
        loss_func=LabelSmoothingCrossEntropy(),
        #callback_fns = [partial(OverSamplingCallback)],  
        metrics=[
            AccumMetric(healthy_roc_auc, flatten=False),
            AccumMetric(multiple_diseases_roc_auc, flatten=False),
            AccumMetric(rust_roc_auc, flatten=False),
            AccumMetric(scab_roc_auc, flatten=False),
            AccumMetric(comp_metric, flatten=False)]
        ).to_fp16()
    return learn
get_learner(fold_num=0).lr_find()
def print_metrics(val_preds, val_labels):
    comp_metric_fold = comp_metric(val_preds, val_labels)
    print(f'Comp metric: {comp_metric_fold}')
    
    healthy_roc_auc_metric = healthy_roc_auc(val_preds, val_labels)
    print(f'Healthy metric: {healthy_roc_auc_metric}')
    
    multiple_diseases_roc_auc_metric = multiple_diseases_roc_auc(val_preds, val_labels)
    print(f'Multi disease: {multiple_diseases_roc_auc_metric}')
    
    rust_roc_auc_metric = rust_roc_auc(val_preds, val_labels)
    print(f'Rust metric: {rust_roc_auc_metric}')
    
    scab_roc_auc_metric = scab_roc_auc(val_preds, val_labels)
    print(f'Scab metric: {scab_roc_auc_metric}')
all_val_preds = []
all_val_labels = []
all_test_preds = []

for i in range(N_FOLDS):
    print(f'Fold {i} results')

    learn = get_learner(fold_num=i)

    #learn.fit_one_cycle(5)
    learn.fit_one_cycle(1)
    learn.unfreeze()

    #learn.fit_one_cycle(6, slice(1e-5, 1e-4))
    learn.fit_one_cycle(1, slice(1e-5, 1e-4))
    
    learn.recorder.plot_loss()
    
    learn.save(f'model_fold_{i}')
    
    learn.freeze()
    
    learn.data = get_data_larger(i)
    
    lr_min,_ = learn.lr_find()
    
    #learn.fit_one_cycle(5, slice(lr_min))
    learn.fit_one_cycle(1, slice(lr_min))
    
    val_preds, val_labels = learn.get_preds()
    
    print_metrics(val_preds, val_labels)
    
    all_val_preds.append(val_preds)
    all_val_labels.append(val_labels)
    
    test_dl = dls.test_dl(test_df)
    test_preds, _ = learn.get_preds(dl=test_dl)
    all_test_preds.append(test_preds)
    
plt.show()
print_metrics(np.concatenate(all_val_preds), np.concatenate(all_val_labels))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15, 10))
interp.plot_confusion_matrix(normalize=True, figsize=(6, 6))
test_df_output = pd.concat([test_df, pd.DataFrame(np.mean(np.stack(all_test_preds), axis=0), columns=LABEL_COLS)], axis=1)
test_df_output.head()
test_df_output.to_csv('submission.csv', index=False)
!head -n 5 submission.csv