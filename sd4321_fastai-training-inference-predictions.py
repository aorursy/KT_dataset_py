!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

import torch

print(torch.__version__)
import fastai

print(fastai.__version__)
from fastai.vision.all import *
path = Path('../input/input/siim-isic-melanoma-classification/')

import pandas as pd
train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

print("Training samples", train_df.shape)

print("First few samples of data are",train_df.head())
train_df.target.value_counts()

train_df_output_1 = train_df[train_df['target']==1]

train_df.shape

train_df_output_0 = train_df[train_df['target']==0].sample(frac=0.03)

train_df_output_0.shape

new_df = pd.concat([train_df_output_0,train_df_output_1]).reset_index(drop=True)

new_df.shape

new_df.head()

new_df.shape, 

new_df.target.value_counts()

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)

fold = 0

for train_index, test_index in skf.split(X=new_df.values, y=new_df.target.values):

    fold+= 1

    print("Fold",fold)

    print("TRAIN LENGTH:", len(train_index), "VALIDATION LENGTH:", len(test_index))

    new_df[f'fold_{fold}_valid']= 0

    new_df.loc[test_index,f'fold_{fold}_valid']= 1

imgpath = Path('../input/siim-isic-melanoma-classification/jpeg/train')

dls = ImageDataLoaders.from_df(new_df, path=imgpath,

                               seed=42, fn_col=0, 

                               suff='.jpg', label_col=7, 

                               valid_col=f'fold_1_valid', item_tfms=Resize(128), 

                               batch_tfms=aug_transforms(flip_vert=True, max_warp=0.), 

                               bs=128, val_bs=None, shuffle_train=True)

print(torch.cuda.is_available())
dls.device
dls.show_batch()

#learn = cnn_learner(dls,resnet34,metrics = [accuracy,roc_auc])

learn = cnn_learner(dls, resnet34, metrics=error_rate)

learn.fine_tune(1)

samplesubmit = "../input/siim-isic-melanoma-classification/sample_submission.csv"

sub = pd.read_csv(samplesubmit)

sub.head()

for i in range(10982):

    x = sub.at[i,'image_name']

    imggpath = Path("../input/siim-isic-melanoma-classification/jpeg/test/" + x + ".jpg")

    pr1,_,pr2 = learn.predict(imggpath)

    sub.at[i,'target'] = float(pr2[int(pr1)])
sub.head()
sub.to_csv('sub1.csv', index=False)