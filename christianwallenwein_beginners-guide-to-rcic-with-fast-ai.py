%reload_ext autoreload

%autoreload 2

%matplotlib inline
import os

import sys

import zipfile



import pandas as pd

import numpy as np



from fastai import *

from fastai.vision import *



from tqdm import tqdm



from pathlib import Path

from PIL import Image
# hide warnings

import warnings

warnings.simplefilter('ignore')
os.listdir("../input")
input_dir = Path("../input/recursion-cellular-image-classification-224-jpg")

input_dir1 = Path("../input/recursion-cellular-image-classification")

print(os.listdir(input_dir))
train_dir = input_dir/"train/train"

test_dir = input_dir/"test/test"

print(os.listdir(train_dir)[:3])

print(os.listdir(test_dir)[:3])
train_csv = pd.read_csv(input_dir/"new_train.csv")

train_csv = train_csv[["filename", "sirna"]]
test_dataset = ImageList.from_folder(input_dir/"test/test")

tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_df(

    path = input_dir,

    df = train_csv,

    folder = "train/train",

    valid_pct = 0.2,

    bs = 32,

    size = 224,

    ds_tfms = tfms,

    num_workers = 0

)

data.add_test(test_dataset)

data.normalize(imagenet_stats)

print(data)

print(len(data.classes))

data.show_batch(rows=3, figsize=(10,10))
learn = cnn_learner(data, models.resnet101, metrics=accuracy, model_dir="/tmp/models")
learn.fit_one_cycle(5)
class_score, y = learn.get_preds(DatasetType.Test)
sample_submission = pd.read_csv(input_dir1/"sample_submission.csv")

sample_submission.head()
class_score
sample_submission.id_code.values[0] in os.listdir(test_dir)
all_test_files = [element[:-8] for element in os.listdir(test_dir)]

all_test_files[:5]
def get_class_score_both(id_code):

    index1 = all_test_files.index(id_code)

    index2 = all_test_files.index(id_code, index1+1)

    sum_class_score = class_score[index1] + class_score[index2]

    avg_class_score = sum_class_score / 2

    return(avg_class_score)
def get_class_score_s1(id_code):

    index1 = all_test_files.index(id_code)

    index2 = all_test_files.index(id_code, index1+1)

    sum_class_score = class_score[index1] + class_score[index2]

    avg_class_score = sum_class_score / 2

    return(avg_class_score)
def get_class_score_s2(id_code):

    index1 = all_test_files.index(id_code)

    index2 = all_test_files.index(id_code, index1+1)

    sum_class_score = class_score[index1] + class_score[index2]

    avg_class_score = sum_class_score / 2

    return(avg_class_score)
x = list(map(get_class_score_both, sample_submission.id_code.values))

x = torch.stack(x)

x = x.argmax(dim=1)

x
submission  = pd.DataFrame({

    "id_code": sample_submission.id_code,

    "sirna": x

})

submission.to_csv("submission.csv", index=False)

submission[:10]