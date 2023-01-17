#Kernel based on https://www.kaggle.com/kenseitrg/simple-fastai-exercise, for my training purpose



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))

from pathlib import Path

from fastai import *

from fastai.vision import *
data_folder = Path("../input")

data_folder.ls()
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/sample_submission.csv")
test_img = ImageList.from_df(test_df, path=data_folder/'test', folder='test')

train_img = (ImageList.from_df(train_df, path=data_folder/'train', folder='train')

        .random_split_by_pct(0.2)

        .label_from_df()

        .add_test(test_img)

        .transform(get_transforms(flip_vert=True), size=128)

        .databunch(path='.', bs=64)

        .normalize(imagenet_stats)

       )
learn = create_cnn(train_img, models.densenet201, metrics=[error_rate, accuracy])
learn.lr_find()
learn.recorder.plot()
lr = 0.1

learn.fit(epochs=3,lr=lr)
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit(epochs=5,lr=2e-6)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(7,6))
preds,_ = learn.get_preds(ds_type=DatasetType.Test)

preds.shape
classes = preds.argmax(1)

classes.shape, classes.min(), classes.max()
test_df.has_cactus = classes

test_df.head()
test_df.to_csv('submission_12_03.csv', index=False)