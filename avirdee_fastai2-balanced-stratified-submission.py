!pip install fastai2 --q
!pip install ../input/fastai017-whl/fastprogress-0.2.3-py3-none-any.whl
!pip install ../input/fastai017-whl/fastcore-0.1.18-py3-none-any.whl
!pip install ../input/fastai017-whl/fastai2-0.0.17-py3-none-any.whl
#Load the dependancies
from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *

import seaborn as sns
import numpy as np
import pandas as pd
import os
import cv2
import openslide

from sklearn.model_selection import StratifiedShuffleSplit

sns.set(style="whitegrid")
sns.set_context("paper")
Path('/root/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)
!cp '../input/resnet18/resnet18.pth' '/root/.cache/torch/checkpoints/resnet18-5c106cde.pth'
source = Path("../input/prostate-cancer-grade-assessment")
files = os.listdir(source)
print(files)
train = source/'train_images'
mask = source/'train_label_masks'
train_labels = pd.read_csv(source/'train.csv')
train_labels.head()
def plot_count(df, feature, title='', size=2):
    f, ax = plt.subplots(1,1, figsize=(3*size,2*size))
    total = float(len(df))
    sns.countplot(df[feature],order = df[feature].value_counts().index, palette='Set1')
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()
plot_count(train_labels, 'isup_grade','ISUP grade - data count and percent', size=3)
isup_0 = train_labels[train_labels.isup_grade == 0]
isup_1 = train_labels[train_labels.isup_grade == 1]
isup_2 = train_labels[train_labels.isup_grade == 2]
isup_3 = train_labels[train_labels.isup_grade == 3]
isup_4 = train_labels[train_labels.isup_grade == 4]
isup_5 = train_labels[train_labels.isup_grade == 5]

print(f'isup_0: {len(isup_0)}, isup_1: {len(isup_1)}, isup_2: {len(isup_2)}, isup_3: {len(isup_3)}, isup_4: {len(isup_4)}, isup_5: {len(isup_5)}')
isup_sam0 = isup_0.sample(n=1224)
isup_sam1 = isup_1.sample(n=1224)
isup_sam2 = isup_2.sample(n=1224)
isup_sam3 = isup_3.sample(n=1224)
isup_sam4 = isup_4.sample(n=1224)

frames = [isup_sam0, isup_sam1, isup_sam2, isup_sam3, isup_sam4, isup_5]
balanced_df = pd. concat(frames)
balanced_df
plot_count(balanced_df, 'isup_grade','ISUP grade - data count and percent', size=3)
df_copy = balanced_df.copy()

# 80/20 split or whatever you choose
train_set = df_copy.sample(frac=0.80, random_state=7)
test_set = df_copy.drop(train_set.index)
print(len(train_set), len(test_set))
#Save train_set to csv for strattification
train_set.to_csv('split.csv', index=False)
df = pd.read_csv('split.csv')
df.shape[0]
sss = StratifiedShuffleSplit(n_splits=6, test_size=0.2, random_state=7)
strat_df = pd.DataFrame()
cols=['isup_grade']

for i, (train_index, test_index) in enumerate(sss.split(df, df.isup_grade)):
    df_split = df.copy()
    df_split['fold'] = i
    df_split.loc[train_index, 'which'] = 'train'
    df_split.loc[test_index, 'which'] = 'valid'
    X_train = df_split.loc[train_index]
    X_valid = df_split.loc[test_index]
    X_train.loc[:, 'which'] = 'train'
    X_valid.loc[:, 'which'] = 'valid'
    
    mult_dis = X_train.loc[X_train.image_id=='isup_grade']
    for _ in range(3): X_train = X_train.append(mult_dis)
    
    strat_df = strat_df.append(X_train).append(X_valid)
    print(i, strat_df.shape, [(X_train[c].sum()/len(X_train), X_train[c].sum()) for c in cols])
strat_df = strat_df.reset_index()
strat_df.head()
def view_image(folder, fn):
    filename = f'{folder}/{fn}.tiff'
    file = openslide.OpenSlide(str(filename))
    t = tensor(file.get_thumbnail(size=(255, 255)))
    pil = PILImage.create(t) 
    return pil
glee_35 = train_labels[train_labels.gleason_score == '3+5']
glee_35[:5]
view_image(train, '05819281002c55258bb3086cc55e3b48')
def get_i(fn):
    filename = f'{train}/{fn.image_id}.tiff'
    example2 = openslide.OpenSlide(str(filename))
    ee = example2.get_thumbnail(size=(255, 255))
    return tensor(ee)
blocks = (
          ImageBlock,
          CategoryBlock
          )    
getters = [
           get_i,
           ColReader('isup_grade')
          ]
trends = DataBlock(blocks=blocks,
              splitter=RandomSplitter(),
              getters=getters,
              item_tfms=Resize(256),
              batch_tfms=aug_transforms()
              )
def train_on_folds(bs, size, base_lr, folds):
    learners = []
    all_val_preds = []
    all_val_labels = []
    all_test_preds = []
    
    for fold in range(folds):
        print(f'Processing fold: {fold}....')
        dls = get_dls(bs=bs, size=size, fold=fold, df=strat_df)
        
        learn = get_learner(dls, arch, loss_func, cbs)
        learn = train_learner(learn, base_lr)
        learn.save(f'model_fold_{fold}')
        learners.append(learn)
        learn.recorder.plot_loss()
        
        test_dl = dls.test_dl(train_df)
        test_preds, _, _ = learn.get_preds(dl=test_dl, with_decoded=True)
        val_preds, val_labels = learn.get_preds()
    
        all_val_preds.append(val_preds)
        all_val_labels.append(val_labels)
        all_test_preds.append(test_preds)
    
    plt.show()
    return learners, all_val_preds, all_val_labels, all_test_preds
set_seed(7)
def get_dls(bs, size, fold, df):
    df_fold = df.copy()
    df_fold = df_fold.loc[df_fold.fold==fold].reset_index()
    
    trends = DataBlock(blocks=blocks,
                       splitter=IndexSplitter(df_fold.loc[df_fold.which=='valid'].index),
                       getters=getters,
                       item_tfms=Resize(256),
                       batch_tfms=aug_transforms(size=size)
                       )
    dls = trends.dataloaders(df_fold, bs=bs)
    assert (len(dls.train_ds) + len(dls.valid_ds)) == len(df_fold)
    return dls
def get_learner(dls,arch,loss_func, cbs):
    return Learner(dls,arch,loss_func=loss_func,metrics=metrics, cbs=cbs).to_fp16()
def train_learner(learn, base_lr):
    learn.unfreeze()
    learn.fit_one_cycle(2, base_lr)
    
    learn.freeze_to(-2)
    learn.fit_one_cycle(2, base_lr)
    return learn
dls = get_dls(bs=32, size=128, fold=2, df=strat_df)
dls.show_batch()
dls.c
arch = resnet18(pretrained=True)
loss_func = LabelSmoothingCrossEntropy(eps=0.3, reduction='mean')
cbs = [ShowGraphCallback()]
metrics=[accuracy,CohenKappa(weights='quadratic')]
train_df = pd.read_csv('split.csv')
bs = 32
size = 196
dls = get_dls(bs=bs, size=size, fold=1, df=strat_df)
learn = Learner(dls,
                model=arch,
                loss_func=loss_func,
                metrics=metrics,
                cbs=cbs)
learn.lr_find()
base_lr = 1e-2
learners, all_val_preds, all_val_labels, all_test_preds = train_on_folds(bs, size, base_lr, folds=1)
test_set
learn.load('model_fold_0')
tst_dl = dls.test_dl(test_set)
tst_dl.show_batch(max_n=9)
_, _, pred_classes = learn.get_preds(dl=tst_dl, with_decoded=True)
pred_classes
test_df = test_set.copy()
test_df['isup_grade_pred'] = pred_classes
test_df
confusion_matrix = pd.crosstab(test_df['isup_grade'], test_df['isup_grade_pred'], rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
plt.show()
from sklearn.metrics import classification_report 
print(classification_report(test_df['isup_grade'], test_df['isup_grade_pred']))

submission_test_path = "../input/prostate-cancer-grade-assessment/test_images/"
sample = '../input/prostate-cancer-grade-assessment/sample_submission.csv'
sub_df = pd.read_csv(sample)
test_df = pd.read_csv(source/f'test.csv')
if os.path.exists(submission_test_path):
    learn.load('prostate')
    def get_inf(df=test_df):
        filename = f'{submission_test_path}/{df.image_id}.tiff' 
        example2 = openslide.OpenSlide(str(filename))
        ee = example2.get_thumbnail(size=(255, 255))
        return tensor(ee)
    
    blocks = (
          ImageBlock,
          CategoryBlock
          )
    getters = [
           get_inf,
           ColReader('isup_grade')
          ]

    trends = DataBlock(blocks=blocks,
              getters=getters,
              item_tfms=Resize(128)
              )
    
    dls = trends.dataloaders(test_df, bs=32)
    learn = cnn_learner(dls, resnet18)
    
    test_dl = dls.test_dl(test_df)
    _,_,pred_classes = learn.get_preds(dl=test_dl, with_decoded=True)
    
    test_df["isup_grade"] = pred_classes
    sub = test_df[["image_id","isup_grade"]]
    sub.to_csv('submission.csv', index=False)
