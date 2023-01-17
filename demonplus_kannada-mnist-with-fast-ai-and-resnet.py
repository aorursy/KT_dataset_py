import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import imageio
path = Path('../input/Kannada-MNIST')

train = pd.read_csv('../input/Kannada-MNIST/train.csv')

test  = pd.read_csv('../input/Kannada-MNIST/test.csv')
train.head()
def to_img_shape(data_X, data_y=[]):

    data_X = np.array(data_X).reshape(-1,28,28)

    data_X = np.stack((data_X,)*3, axis=-1)

    data_y = np.array(data_y)

    return data_X,data_y
data_X, data_y = train.loc[:,'pixel0':'pixel783'], train['label']



from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(data_X, data_y, test_size=0.1,random_state=42,stratify=data_y)



test_X = test.loc[:,'pixel0':'pixel783']
train_X,train_y = to_img_shape(train_X, train_y)

val_X,val_y = to_img_shape(val_X,val_y)

test_X, _ = to_img_shape(test_X)
def save_imgs(path:Path, data, labels):

    path.mkdir(parents=True,exist_ok=True)

    for label in np.unique(labels):

        (path/str(label)).mkdir(parents=True,exist_ok=True)

    for i in range(len(data)):

        if(len(labels)!=0):

            imageio.imsave( str( path/str(labels[i])/(str(i)+'.jpg') ), data[i] )

        else:

            imageio.imsave( str( path/(str(i)+'.jpg') ), data[i] )



save_imgs(Path('/data/train'),train_X,train_y)

save_imgs(Path('/data/valid'),val_X,val_y)

save_imgs(Path('/data/test'),test_X, [])
tfms = get_transforms(do_flip=False, max_rotate=10, max_zoom=1.1, max_warp=0.0, max_lighting=0.1)



data = (ImageList.from_folder('/data/') 

        .split_by_folder()          

        .label_from_folder()        

        .add_test_folder()          

        .transform(tfms, size=64)   

        .databunch())
data.show_batch(3,figsize=(6,6))
data.show_batch(3,figsize=(6,6), ds=DatasetType.Test)
!mkdir -p /tmp/.cache/torch/checkpoints

!cp /kaggle/input/pretrained-pytorch/resnet50-19c8e357.pth /tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth



learn = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir = Path('../kaggle/working'),path = Path("."))
learn.lr_find()

learn.recorder.plot()
lr = 1e-2

learn.fit_one_cycle(4)
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
lr = 1e-4

learn.fit_one_cycle(4,lr)
!cp /kaggle/input/pretrained-pytorch/resnet101-5d3b4d8f.pth /tmp/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth



learn101 = cnn_learner(data, models.resnet101, metrics=accuracy, model_dir = Path('../kaggle/working'),path = Path("."))
learn101.lr_find()

learn101.recorder.plot()
lr = 1e-2

learn101.fit_one_cycle(4)
learn101.unfreeze()
learn101.lr_find()

learn101.recorder.plot()
lr = 1e-4

learn101.fit_one_cycle(4,lr)
!cp /kaggle/input/pretrained-pytorch/resnet152-b121ed2d.pth /tmp/.cache/torch/checkpoints/resnet152-b121ed2d.pth



learn152 = cnn_learner(data, models.resnet152, metrics=accuracy, model_dir = Path('../kaggle/working'),path = Path("."))
learn152.lr_find()

learn152.recorder.plot()
lr = 1e-2

learn152.fit_one_cycle(4)
learn152.unfreeze()
learn152.lr_find()

learn152.recorder.plot()
lr = 1e-4

learn152.fit_one_cycle(4,lr)
preds50, _ = learn.get_preds(DatasetType.Test)

preds101, _ = learn101.get_preds(DatasetType.Test)

preds152, _ = learn152.get_preds(DatasetType.Test)



preds = 0.34*preds50 + 0.33*preds101 + 0.33*preds152



y = torch.argmax(preds, dim=1)
num = len(learn.data.test_ds)

indexes = {}



for i in range(num):

    filename = str(learn.data.test_ds.items[i]).split('/')[-1]

    filename = filename[:-4] # get rid of .jpg

    indexes[(int)(filename)] = i
submission = pd.DataFrame({ 'id': range(0, num),'label': [y[indexes[x]].item() for x in range(0, num)] })

submission.to_csv(path_or_buf ="submission.csv", index=False)