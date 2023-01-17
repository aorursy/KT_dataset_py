# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
import matplotlib.pyplot as plt

import os



from fastai.vision import *

import torchvision.models as models
!ls ../input
all_train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print(all_train_df.shape, test_df.shape)
all_train_df.head()
val_df = all_train_df.sample(frac=0.2, random_state=1337)

train_df = all_train_df.drop(val_df.index)

train_df['fn'] = train_df.index

print(train_df.shape, val_df.shape)
train_df.head()
class PixelImageItemList(ImageList):

    def open(self,fn):

        regex = re.compile(r'\d+')

        fn = re.findall(regex,fn)

        df = self.inner_df[self.inner_df.fn.values == int(fn[0])]

        df_fn = df[df.fn.values == int(fn[0])]

        img_pixel = df_fn.drop(labels=['label','fn'],axis=1).values

        img_pixel = img_pixel.reshape(28,28)

        img_pixel = np.stack((img_pixel,)*3,axis=-1)

        return vision.Image(pil2tensor(img_pixel,np.float32).div_(255))
src = (PixelImageItemList.from_df(train_df,'.',cols='fn')

      .split_by_rand_pct()

      .label_from_df(cols='label'))
data = (src.transform(tfms=(rand_pad(padding=5,size=28,mode='zeros'),[]))

       .databunch(num_workers=2,bs=128))
data
arch = vision.models.resnet152

learn = cnn_learner(data, arch, pretrained=True, metrics=accuracy)
learn.lr_find()
learn.recorder.plot()
lr = 5e-2

learn.fit_one_cycle(3, slice(lr))
learn.save('frozen-resnet50')

learn.unfreeze()
learn.recorder.plot_losses()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, slice(5e-5, lr/10))
learn.recorder.plot_losses()
df_test = pd.read_csv('../input/test.csv')

df_test['label'] = 0

df_test['fn'] = df_test.index

df_test.head()
learn.data.add_test(PixelImageItemList.from_df(df_test, path='.', cols='fn'))

test_pred, test_y, test_loss = learn.get_preds(ds_type=DatasetType.Test, with_loss=True)

test_result = torch.argmax(test_pred,dim=1)

result = test_result.numpy()
test_pred.shape

#create a CSV file to submit

final = pd.Series(result,name='Label')

submission = pd.concat([pd.Series(range(1,28001),name='ImageId'),final],axis=1)

submission.to_csv('submission.csv',index=False)
!ls