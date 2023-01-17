# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from fastai.vision import *
import pandas as pd

df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df.head()
path = '/kaggle/input/digit-recognizer'
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
        df = np.array(df)/255.
        res.items = (df-df.mean())/df.std()

        return res
tfms = get_transforms(do_flip=False)
src = (NumpyImageList.from_csv(path, 'train.csv')
        .split_by_rand_pct(.2)
        .label_from_df(cols='label'))
data = (src.transform(tfms, size=28)
        .databunch().normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(12,9))

learn = cnn_learner(data, models.resnet18, metrics=[accuracy],model_dir = '/kaggle/working')
learn.lr_find()
learn.recorder.plot(suggestion=True)
lr = learn.recorder.min_grad_lr
learn.fit_one_cycle(5,lr)
learn.save('stage1')
learn.unfreeze()
data = (src.transform(tfms, size=64)
        .databunch().normalize(imagenet_stats))

learn.data = data
data.train_ds[0][0].shape
learn.lr_find()
learn.recorder.plot(suggestion=True)
lr = learn.recorder.min_grad_lr
learn.fit_one_cycle(5,lr)
learn.save('stage-1-64')

learn = learn.load("stage-1-64")
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(6,6))
interp.plot_confusion_matrix()
test =  (NumpyImageList.from_csv(path, 'test.csv'))
print(len(test))
data.add_test(test) 
predictions, y, losses = learn.get_preds(ds_type=DatasetType.Test, with_loss=True)

labels = np.argmax(predictions, 1)
len(labels)
submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})
submission_df.head()
submission_df.to_csv(f'submission.csv', index=False)
len(submission_df)
