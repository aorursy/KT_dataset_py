import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
!pip install iterative-stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# !pip install fastai==2

!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()


import torch
torch.cuda.is_available()    # Just realized cuda is unavailable. What a waste of GPU.
import fastai
fastai.__version__
# !pip install fastcore==1.0.0
from torchvision.models import densenet121
import os
import random


from fastbook import *
import fastai
from fastai.vision.widgets import *
from sklearn.metrics import roc_auc_score
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
DATA_PATH = Path("../input/plant-pathology-2020-fgvc7")
IMG_PATH = DATA_PATH/"images"
SEED = 42
LABEL_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

IMG_SIZE = 512
BATCH_SIZE = 16
N_FOLDS = 5

ARCH = densenet121     # A smaller model than Resnet with good performance

seed_everything(SEED)
train_df = pd.read_csv(DATA_PATH/"train.csv")
test_df = pd.read_csv(DATA_PATH/"test.csv")
train_df.shape, test_df.shape
train_df.head()
train_df.info()       # No missing values
train_df.iloc[:,1:].sum().plot.bar()
train_df[['healthy', 'multiple_diseases', 'rust', 'scab']].sum(axis=1).unique()
train_df["fold"] = -1

strat_kfold = MultilabelStratifiedKFold(n_splits = N_FOLDS, random_state = SEED, shuffle = True)
for i, (_,test_index) in enumerate(strat_kfold.split(train_df["image_id"].values, train_df.iloc[:, 1:].values)):
    train_df.iloc[test_index, -1] = i
    
train_df["fold"] = train_df["fold"].astype('int')
train_df
fig, axes = plt.subplots(nrows=1, ncols = 5, constrained_layout = True, figsize=(15,3), sharey=True)
for ax, fold in zip(axes, range(5)):
    train_df[train_df["fold"] == fold].iloc[:,1:-1].sum().plot.bar(ax = ax)
    ax.set_title(f'Fold {fold} label dist')
# Next steps create DataBlocks, Oversampling, create learners, write metrics, training
train_df['label'] = train_df[LABEL_COLS].idxmax(axis = 1)
train_df.head()
train_df['label'].value_counts()
# This function upsample the 'multiple_diseases' label

def upsample(fold):
    train_df_no_val = train_df[train_df['fold'] != fold]     # This is the training data
    train_df_just_val = train_df[train_df['fold'] == fold]   # This is the validation data. This method will prevent overfitting
    
    train_df_bal = pd.concat(
                    [train_df_no_val[train_df_no_val['label'] != 'multiple_diseases'], train_df_just_val] + 
                    [train_df_no_val[train_df_no_val['label'] == 'multiple_diseases']] * 3).sample(frac = 1.0, random_state = SEED).reset_index(drop = True)
        
    return train_df_bal

upsample(fold = 0)['label'].value_counts()
# This function returns a dataloader object

def get_data(fold):
    data = upsample(fold = fold)
    
    datablock = DataBlock(
                blocks = (ImageBlock, CategoryBlock(vocab = LABEL_COLS)),
                getters = [
                    ColReader('image_id', pref = IMG_PATH, suff = '.jpg'),
                    ColReader('label')
                ],
                splitter = IndexSplitter(data.loc[data['fold'] == fold].index),
                item_tfms = Resize(IMG_SIZE),
                batch_tfms = aug_transforms(size = IMG_SIZE, max_rotate = 30., min_scale = 0.75, flip_vert = True, do_flip = True))
    
    return datablock.dataloaders(source = data, bs = BATCH_SIZE)
    
dls = get_data(fold = 0)
dls.show_batch()
def comp_metrics(preds, targs, labels = range(len(LABEL_COLS))):
    # Average of individual disease auc roc
    targs = np.eye(4)[targs]     
    return np.mean([roc_auc_score(targs[:,i], preds[:,i]) for i in labels])

def healthy_roc_auc(*args):
    return comp_metrics(*args, labels = [0])

def multiple_diseases_roc_auc(*args):
    return comp_metrics(*args, labels = [1])

def rust_roc_auc(*args):
    return comp_metrics(*args, labels = [2])


def scab_roc_auc(*args):
    return comp_metrics(*args, labels = [3])
def get_learner(fold_num, lr = 1e-3):
    opt_func = partial(Adam, lr = lr, wd = 0.01, eps = 1e-8)  # Optimizer... Not sure what is partial
    
    data = get_data(fold_num)
    
    learn = cnn_learner(
            data, ARCH, opt_func = opt_func,
            loss_func = LabelSmoothingCrossEntropy(),   # Helps the model to train around mislabeled data,better performance, robustness
            metrics = [
                AccumMetric(healthy_roc_auc, flatten = False),
                AccumMetric(multiple_diseases_roc_auc, flatten = False),
                AccumMetric(rust_roc_auc, flatten = False),
                AccumMetric(scab_roc_auc, flatten = False),
                AccumMetric(comp_metrics, flatten = False)
            ]).to_fp16()                                  # Lower fixed precision, low memory, good performance
    
    return learn
get_learner(fold_num=0).lr_find()
def print_metrics(val_preds, val_labels):
    print("Comp Metric: ", comp_metrics(val_preds, val_labels))
    print("Healthy Metric: ", healthy_roc_auc(val_preds, val_labels))
    print("Multi diseases: ", multiple_diseases_roc_auc(val_preds, val_labels))
    print("Rust Metric: ", rust_roc_auc(val_preds, val_labels))
    print("Scab Metric: ", scab_roc_auc(val_preds, val_labels))
all_val_preds = []
all_val_labels = []
all_test_preds = []

for i in range(N_FOLDS):
    print("Fold {} RESULT".format(i))
    
    learn = get_learner(i)
    learn.fit_one_cycle(4)
    learn.unfreeze()
    learn.fit_one_cycle(6, slice(1e-5, 1e-4))
    learn.recorder.plot_loss()
    
    learn.save(f"model_fold_{i}")
    val_preds, val_labels = learn.get_preds()
    print_metrics(val_preds, val_labels)
    all_val_preds.append(val_preds)
    all_val_labels.append(val_labels)
    
    test_dl = dls.test_dl(test_df)
    test_preds, _ = learn.get_preds(dl=test_dl)
    all_test_preds.append(test_preds)
    
plt.show()
print_metrics(np.concatenate(all_val_preds), np.concatenate(all_val_labels))
# Identify images with top losses

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(normalize=True, figsize=(6, 6))
interp.plot_confusion_matrix(figsize=(6, 6))
interp.plot_top_losses(9, figsize=(15, 10))
