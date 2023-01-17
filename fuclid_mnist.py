# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline







# Any results you write to the current directory are saved as output.
# Load the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head(2)

test.head(2)
from fastai.vision import *

from fastai.metrics import error_rate
class CustomImageItemList(ImageItemList):

    def open(self, fn):

        img = fn.reshape(28,28)

        img = np.stack((img,)*3, axis=-1) # convert to 3 channels

        img = img/255.0

        return Image(pil2tensor(img, dtype=np.float32))



    @classmethod

    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList':

        df = pd.read_csv(Path(path)/csv_name, header=header)

        res = super().from_df(df, path=path, cols=0, **kwargs)

        # convert pixels to an ndarray

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values, axis=1).values

        return res
doc(get_transforms)
tfms = get_transforms(do_flip=False)
test = CustomImageItemList.from_csv_custom(path="../input", csv_name='test.csv', imgIdx=0)



data = (CustomImageItemList.from_csv_custom(path="../input", csv_name='train.csv')

                       .random_split_by_pct(.2)

                       .label_from_df(cols='label')

                       .add_test(test, label=0)

                       .databunch(bs=64, num_workers=0))
print(data.train_ds.x[0])

a = data.train_ds.x[0]

print(a.data.reshape(-1).numpy())

print(np.max(a.data.reshape(-1).numpy(),0))
learn = create_cnn(data, models.resnet18, metrics=accuracy,path = '')
data.show_batch(rows=3, figsize=(5,5))
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4,max_lr=slice(1.5e-02))
learn.lr_find()

learn.recorder.plot()
1.5e-02
learn.fit_one_cycle(4,max_lr=slice(3e-6,0.015/10))
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6,max_lr=slice(1e-6,1e-5))
preds, y = learn.get_preds(DatasetType.Test)
print(preds[0])
results = np.argmax(preds,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)