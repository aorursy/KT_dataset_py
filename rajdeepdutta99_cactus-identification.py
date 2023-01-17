import os

import numpy as np

import pandas as pd

from fastai.vision import *

from fastai.callbacks import *
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!unzip /kaggle/input/aerial-cactus-identification/train.zip
!unzip /kaggle/input/aerial-cactus-identification/test.zip
data = ImageDataBunch.from_csv(path = "/kaggle",

                               folder = "working/train/",

                               csv_labels = "input/aerial-cactus-identification/train.csv",

                               seed = 42,

                               test = "working/test/",

                               bs = 1024,

                               ds_tfms = get_transforms(),

                               size = 32,

                               num_workers = 6).normalize(imagenet_stats)

data
print(data.classes)

print(data.c)
data.show_batch(3, figsize=(6,6))
learn = cnn_learner(data, models.resnet50, metrics = [accuracy, AUROC()])
learn.fit_one_cycle(8)
learn.save("model")
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(16, max_lr = slice((1e-5)/2, 1e-4), callbacks = [SaveModelCallback(learn, name = "best_finetuned_model")])
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
test_preds, _ = learn.get_preds(DatasetType.Test)

print(len(test_preds))

test_preds
test_preds[:,1]
test_files = data.test_dl.items

test_files
dic = {'id': test_files, 'has_cactus':test_preds[:,1]}

dic
df = pd.DataFrame(dic)

df
df['id'] = pd.Series(str(df['id'][i]).split("/")[-1] for i in range(df.shape[0]))

df
df = df.sort_values(by=['id'], axis=0).reset_index(drop=True)

df
df.to_csv("test_output.csv", index = False)