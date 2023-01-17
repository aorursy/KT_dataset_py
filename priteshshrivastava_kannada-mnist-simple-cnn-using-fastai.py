%reload_ext autoreload

%autoreload 2

%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from fastai.vision import *

from fastai.tabular import *

import re



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path = '/kaggle/input/Kannada-MNIST/'

train = pd.read_csv(path+'train.csv')

train.head()
test = pd.read_csv(path+'test.csv')

test.head()
class NumpyImageList(ImageList):

    def open(self, fn):

        img = fn.reshape(28,28,1)

        return Image(pil2tensor(img, dtype=np.float32))

    

    @classmethod

    def from_csv(cls, path:PathOrStr, csv:str, **kwargs)->'ItemList': 

        df = pd.read_csv(Path(path)/csv, header='infer')

        res = super().from_df(df, path=path, cols=0, **kwargs)



        if 'label' in df.columns:

            df = df.drop('label', axis=1)

        if 'id' in df.columns:

            df = df.drop('id', axis=1)

        df = np.array(df)/255.

        res.items = (df-df.mean())/df.std()



        return res

    

defaults.cmap='binary'
test = NumpyImageList.from_csv(path, 'test.csv')

test
train = NumpyImageList.from_csv(path, 'train.csv')

train
tfms = get_transforms(do_flip=False)

data = (NumpyImageList.from_csv(path, 'train.csv')

        .split_by_rand_pct(.1)

        .label_from_df(cols='label')

        .add_test(test, label=0)

        .transform(tfms)

        .databunch(bs=128, num_workers=0)

        .normalize(imagenet_stats))

data
data.show_batch(rows=5, figsize=(10,10))
xb,yb = data.one_batch()

xb.shape,yb.shape
def conv(ni,nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)



model = nn.Sequential(

    conv(3, 8), # 14   ## Why not 1 ??

    nn.BatchNorm2d(8),

    nn.ReLU(),

    conv(8, 16), # 7

    nn.BatchNorm2d(16),

    nn.ReLU(),

    conv(16, 32), # 4

    nn.BatchNorm2d(32),

    nn.ReLU(),

    conv(32, 16), # 2

    nn.BatchNorm2d(16),

    nn.ReLU(),

    conv(16, 10), # 1

    nn.BatchNorm2d(10),

    Flatten()     # remove (1,1) grid

)



if torch.cuda.is_available():

    model = model.cuda()

learn = Learner(data, model, metrics=accuracy, model_dir='/kaggle/working/models')

learn.summary()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=slice(1e+0))
predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)

# output to a file

submission_df = pd.DataFrame({'id': list(range(0,len(labels))), 'label': labels})

submission_df.head()
## Check format of sample submission

sample_submission = pd.read_csv(path+'sample_submission.csv')

sample_submission.head()
## Our submission files matches the required format, so we can submit it

submission_df.to_csv('submission.csv', index=False)