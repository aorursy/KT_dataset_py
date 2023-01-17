%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

from pathlib import Path

import os

import pandas as pd
print(os.listdir("../input"))
train_dir = "../input/train/train/"

test_dir = "../input/test/test/"
print(os.listdir(train_dir)[:5])

print(os.listdir(test_dir)[:5])
train_csv = pd.read_csv("../input/train.csv")

train_csv.head()
tfms = get_transforms()
data = ImageDataBunch.from_df(

    df = train_csv,

    path = train_dir,

    test = "../../test",

    valid_pct = 0.2,

    bs = 16,

    size = 32,

    ds_tfms = tfms,

    num_workers = 0

).normalize(imagenet_stats)

print(data.classes)

data.show_batch(rows=2)
learn = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/tmp/models")
learn.fit_one_cycle(1)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_multi_top_losses(2, figsize=(6, 6))
interp.plot_confusion_matrix()
class_score, y = learn.get_preds(DatasetType.Test)

class_score = np.argmax(class_score, axis=1)
submission  = pd.DataFrame({

    "id": os.listdir(test_dir),

    "has_cactus": class_score

})

submission.to_csv("submission.csv", index=False)

submission[:5]