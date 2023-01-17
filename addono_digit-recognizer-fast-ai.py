%reload_ext autoreload

%autoreload 2

%matplotlib inline



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from fastai.vision import *

from fastai.metrics import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

path = '../input'

print(os.listdir(path))



# Any results you write to the current directory are saved as output.
class CustomImageList(ImageList):

    def open(self, fn):

        img = fn.reshape(28,28)

        img = np.stack((img,)*3, axis=-1)

        return Image(pil2tensor(img, dtype=np.float32))

    

    @classmethod

    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList': 

        df = pd.read_csv(Path(path)/csv_name, header=header)

        res = super().from_df(df, path=path, cols=0, **kwargs)

        

        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values

        

        return res
test = CustomImageList.from_csv_custom(path=path, csv_name='test.csv', imgIdx=0)
tfms = get_transforms(do_flip=False)

data = (CustomImageList.from_csv_custom(path=path, csv_name='train.csv', imgIdx=1)

                .split_by_rand_pct(.2)

                .label_from_df(cols='label')

                .add_test(test, label=0)

                .transform(tfms)

                .databunch(bs=128, num_workers=0)

                .normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(5,5))
learn = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir='/kaggle/working/models')
learn.lr_find()
learn.recorder.plot()
lr = 1e-2

learn.fit_one_cycle(10, lr)
learn.save('stage1-resnet50')
learn.load('stage1-resnet50')

learn.validate()
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(5, max_lr=slice(1e-6, 2.5e-4))
learn.validate()
# get the predictions

predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)

# output to a file

submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})

submission_df.to_csv(f'submission.csv', index=False)