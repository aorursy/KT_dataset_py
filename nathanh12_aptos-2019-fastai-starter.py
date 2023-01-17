import numpy as np

import pandas as pd



import os

print(os.listdir("../input"))



from pathlib import Path

from fastai import *

from fastai.vision import *
data_folder = Path("../input/")
data_folder.ls()
train_df = pd.read_csv(data_folder/'train.csv')

test_df = pd.read_csv(data_folder/'sample_submission.csv')
test_data = ImageList.from_df(test_df, path=data_folder, folder='test_images', suffix='.png')
data = (ImageList.from_df(train_df, path=data_folder, folder='train_images', suffix = '.png')

                 .split_by_rand_pct(0.1)  

                 .label_from_df()

                 .add_test(test_data)              

                 .transform(get_transforms(flip_vert=True), size=128)

                 .databunch(path='.', bs=32)

                 .normalize(imagenet_stats)

       )
data.show_batch(3)
learner = cnn_learner(data, models.resnet18, pretrained=False, metrics=[accuracy])
learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(5, 1e-3)
learner.recorder.plot_losses()

learner.recorder.plot_metrics()
learner.unfreeze()
learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(10, 1e-3)
learner.recorder.plot_losses()

learner.recorder.plot_metrics()
preds,_ = learner.get_preds(ds_type=DatasetType.Test)
test_df.diagnosis = preds.argmax(1)

test_df.head()
test_df.to_csv('submission.csv', index=False)