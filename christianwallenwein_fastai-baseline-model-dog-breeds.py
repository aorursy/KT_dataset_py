%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import os

import pandas as pd

from pathlib import Path
input_dir = Path("../input")

print(os.listdir(input_dir))
train_dir = input_dir/'train'

test_dir = input_dir/'test'
print(os.listdir(train_dir)[:5])

print(os.listdir(test_dir)[:5])
labels_path = "../input/labels.csv"

labels_csv = pd.read_csv(labels_path)

labels_csv.head()
tfms = get_transforms()

data = ImageDataBunch.from_csv(

    path = "../input",

    folder = "train",

    suffix = ".jpg",

    test = "test/test",

    bs = 16,

    size = 224,

    ds_tfms = tfms,

    num_workers = 0

).normalize(imagenet_stats)

print(data.classes[:10])

data.show_batch(rows=2)
learn = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/tmp/models")
learn.fit_one_cycle(1)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_multi_top_losses(4, figsize=(10, 10))
interp.most_confused(min_val=5)
class_score, y = learn.get_preds(DatasetType.Test)
# let's first check the sample submission file to understand the format of the submission file

sample_submission =  pd.read_csv(input_dir/"sample_submission.csv")

display(sample_submission.head(3))
classes_series = pd.Series(os.listdir(test_dir))

classes_series = classes_series.str[:-4]

classes_df = pd.DataFrame({'id':classes_series})

predictions_df = pd.DataFrame(class_score.numpy(), columns=data.classes)

submission = pd.concat([classes_df, predictions_df], axis=1)
submission.to_csv("submission.csv", index=False)

submission[:5]