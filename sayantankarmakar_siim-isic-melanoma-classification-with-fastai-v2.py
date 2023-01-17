!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

!pip install --upgrade fastai
import torch

import torch.nn as nn

from fastai.vision.all import * 
!nvidia-smi
torch.__version__
torch.cuda.is_available()
path = Path('/kaggle/input/siim-isic-melanoma-classification/')

path.ls()
train_df = pd.read_csv(path/'train.csv')

test_df = pd.read_csv(path/'test.csv')

print("train_df: ", train_df.shape)

print("test_df: ", test_df.shape)
train_df.head()
from sklearn.model_selection import StratifiedKFold



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



for fold, (t_, v_) in enumerate(skf.split(X=train_df.values, y=train_df.target.values)):

    train_df.loc[v_, 'fold'] = fold
for fold in range(5):

    print("fold:", fold, end=" - ")

    print(len(train_df[train_df['fold'] == fold]))
def get_x(df):

    image_name = df['image_name']

    return path/'jpeg'/'train'/f'{image_name}.jpg'

def get_y(df):

    return df['target']

def splitter(df, fold=0):

    train = df.index[df.fold != fold].tolist()

    valid = df.index[df.fold == fold].tolist()

    return train, valid
dblock = DataBlock(

    blocks=(ImageBlock, CategoryBlock),

    get_x=get_x,

    get_y=get_y,

    splitter=RandomSplitter(),

    item_tfms=RandomResizedCrop(128, min_scale=0.35),

)

dls = dblock.dataloaders(train_df)
dls.show_batch(nrows=1, ncols=3)
dls.device
learn = cnn_learner(dls, resnet18, metrics=accuracy)
learn.fine_tune(1)