# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%reload_ext autoreload

%autoreload 2

%matplotlib inline



import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        

    # Any results you write to the current directory are saved as output.

    

from fastai import *

from fastai.vision import *

from fastai.core import *
import torch

torch.cuda.device(0) 

torch.backends.cudnn.benchmark=True

path = Path('../input/digit-recognizer')
class CustomImageItemList(ImageList):

    def open(self, fn):

        img = fn.reshape(28, 28)

        img = np.stack((img,)*3, axis=-1) # convert to 3 channels

        return Image(pil2tensor(img, dtype=np.float16))



    @classmethod

    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs) -> 'ItemList':

        df = pd.read_csv(Path(path)/csv_name, header=header)

        res = super().from_df(df, path=path, cols=0, **kwargs)

        # convert pixels to an ndarray

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values

        return res
test = CustomImageItemList.from_csv_custom(path=path, csv_name='test.csv', imgIdx=0)

data = (CustomImageItemList.from_csv_custom(path=path, csv_name='train.csv')

                       .split_by_rand_pct(.25)

                       .label_from_df(cols='label')

                       .add_test(test, label=0)

                       .databunch(bs=64, num_workers=0)

                       .normalize(imagenet_stats))
learn = cnn_learner(data, models.resnet50, metrics=accuracy,model_dir='/tmp/models')

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, slice(5*1e-04,0.2))

learn.save('stage-1')

learn.load('stage-1')

learn.recorder.plot_losses()

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9,figsize=(9,9))

interp.plot_confusion_matrix(figsize=(8,8))
learn.unfreeze()

# learn.to_fp32()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=slice(1e-5,1e-3))

learn.recorder.plot_losses()

learn.save('stage-2')
learn.recorder.plot_losses()

learn.load('stage-2')

preds, y, losses = learn.get_preds(ds_type=DatasetType.Test, with_loss=True)
y = torch.argmax(preds, dim=1)



submission_df = pd.DataFrame({'ImageId': range(1, len(y) + 1), 'Label': y}, columns=['ImageId', 'Label'])

submission_df.head()
submission_df.to_csv('submission.csv', index=False)

!kaggle competitions submit -c digit-recognizer -f submission.csv -m "Message"