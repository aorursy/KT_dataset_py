%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai import *

from fastai.metrics import error_rate

import pandas as pd

import torch
path ="../input/"

train_df=pd.read_csv(path+"train.csv")

test_df=pd.read_csv(path+"sample_submission.csv")
bs = 128

data = ImageDataBunch.from_csv(path=path, folder='train/train', csv_labels='train.csv', ds_tfms=get_transforms(), size=32, bs=bs).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.densenet161, metrics=error_rate, model_dir="/tmp/model/")
learn.fit_one_cycle(6)
learn.save('stage-1')

learn.unfreeze()

learn.lr_find()

learn.recorder.plot()

learn.fit_one_cycle(5, max_lr=slice(3e-5,3e-4))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
a,b,c=learn.predict(open_image("../input/test/test/"+"000940378805c44108d287872b2f04ce.jpg"))

print(c)

print(c[1].numpy())
test_df.head()
def pred(name):

    a,b,c=learn.predict(open_image("../input/test/test/"+name))

    return c[1].numpy()
test_df["has_cactus"]=test_df["id"].apply(lambda x:pred(x))
test_df.to_csv('submission.csv',index=False)