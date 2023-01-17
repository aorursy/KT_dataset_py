# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai import *

from fastai.vision import *

from fastai.callbacks import * 

import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision import models as visonmodels



# Any results you write to the current directory are saved as output.
!ls ../input

!mkdir -p /tmp/.torch/models/

!cp ../input/pretrained-pytorch-models/* /tmp/.torch/models/

!cp ../input/resnet-from-fastai/* /tmp/.torch/models
path = Path('../input/digit-recognizer')
class CustomImageItemList(ImageList):

    def open(self, fn):

        img = fn.reshape(28, 28)

        img = np.stack((img,)*3, axis=-1) # convert to 3 channels

        return Image(pil2tensor(img, dtype=np.float32))



    @classmethod

    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs) -> 'ItemList':

        df = pd.read_csv(Path(path)/csv_name, header=header)

        res = super().from_df(df, path=path, cols=0, **kwargs)

        # convert pixels to an ndarray

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 783.0, axis=1).values

        return res
test = CustomImageItemList.from_csv_custom(path=path, csv_name='test.csv', imgIdx=0)

data = (CustomImageItemList.from_csv_custom(path=path, csv_name='train.csv')

                       .split_by_rand_pct(.2)

                       .label_from_df(cols='label')

                       .add_test(test, label=0)

                       .databunch(bs=64, num_workers=0)

                       .normalize(imagenet_stats))
x, y = next(iter(data.train_dl))

print(x.shape)

print(y.shape)
data.show_batch(rows=3, figsize=(12,9))
arch1 = models.resnet34
arch1()
class Net1(nn.Module):

    def __init__(self,pretrained):

        super(Net1, self).__init__()

        self.conv1 = nn.Conv2d(3, 20, 5, 1)

        self.conv2 = nn.Conv2d(20, 50, 5, 1)

        self.fc1 = nn.Linear(4*4*50, 500)

        self.fc2 = nn.Linear(500, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 4*4*50)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
# class Net1(nn.Module):

#     def __init__(self,pretrained):

#         super(Net1, self).__init__()

#         self.body=nn.Sequential(

#             nn.Conv2d(3, 20, 5, 1),

#             nn.ReLU(),

#             nn.MaxPool2d(2),

#             nn.Conv2d(20, 50, 5, 1),

#             nn.ReLU(),

#             nn.MaxPool2d(2),



#         )

#         self.head=nn.Sequential(

#             nn.Linear(4*4*50, 500),

#             nn.ReLU(),

#             nn.Linear(500, 10),

#             nn.LogSoftmax(dim=1)

#         )



#     def forward(self, x):

#         x=self.body(x)

#         x = x.view(-1, 4*4*50)

#         return self.head(x)
arch2 = Net1
print(torch.__version__)
torch.cuda.is_available()
learn = Learner(data, arch2(False),metrics=error_rate,model_dir='/tmp/models')
# learn =  cnn_learner(data, arch2,pretrained=False,model_dir='/tmp/models')
learn.lr_find()

learn.recorder.plot()
lr = 0.03
learn.fit_one_cycle(3, slice(lr))
# learn.fit_one_cycle(30, max_lr=slice(3e-5,3e-4),callbacks=[SaveModelCallback(learn,monitor='error_rate',mode='min')])

#rule of thumb do Lowest error rate.
learn.recorder.plot_losses()
learn.save('stage-1')
learn.load('stage-1')
interp = learn.interpret()
interp.plot_top_losses(9, figsize=(7,7))
preds, y, losses = learn.get_preds(ds_type=DatasetType.Test, with_loss=True)
# Bug in fastai? Why is this needed?

y = torch.argmax(preds, dim=1)
submission_df = pd.DataFrame({'ImageId': range(1, len(y) + 1), 'Label': y}, columns=['ImageId', 'Label'])

submission_df.head()
submission_df.to_csv('submission.csv', index=False)
!head submission.csv