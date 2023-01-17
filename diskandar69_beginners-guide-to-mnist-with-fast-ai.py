%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import os

import pandas as pd

import numpy as np

from pathlib import Path

from PIL import Image
# hide warnings

import warnings

warnings.simplefilter('ignore')
input_dir = Path("../input/digit-recognizer")

os.listdir(input_dir)
train_df =  pd.read_csv(input_dir/"train.csv")

train_df.head(3)
test_df =  pd.read_csv(input_dir/"test.csv")

test_df.head(3)
train_dir = Path("../train")

test_dir = Path("../test")
# Create training directory

for index in range(10):

    try:

        os.makedirs(train_dir/str(index))

    except:

        pass
# Test whether creating the training directory was successful

sorted(os.listdir(train_dir))
#Create test directory

try:

    os.makedirs(test_dir)

except:

    pass
# save training images

for index, row in train_df.iterrows():

    

    label,digit = row[0], row[1:]

    

    filepath = train_dir/str(label)

    filename = f"{index}.jpg"

    

    digit = digit.values

    digit = digit.reshape(28,28)

    digit = digit.astype(np.uint8)

    

    img = Image.fromarray(digit)

    img.save(filepath/filename)
# save testing images

for index, digit in test_df.iterrows():



    filepath = test_dir

    filename = f"{index}.jpg"

    

    digit = digit.values

    digit = digit.reshape(28,28)

    digit = digit.astype(np.uint8)

    

    img = Image.fromarray(digit)

    img.save(filepath/filename)
tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_folder(

    path = train_dir,

    test = test_dir,

    valid_pct = 0.2,

    bs = 32,

    size = 28,

    ds_tfms = tfms,

    num_workers = 0

).normalize(imagenet_stats)

print(data)

print(data.classes)

data.show_batch(figsize=(5,5))
learn = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/tmp/models")
learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(7, 7))
interp.plot_confusion_matrix()
#let's unfreeze the whole model!

learn.unfreeze()
learn.fit_one_cycle(1)
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))
class_score, y = learn.get_preds(DatasetType.Test)

class_score = np.argmax(class_score, axis=1)
sample_submission =  pd.read_csv(input_dir/"sample_submission.csv")

display(sample_submission.head(2))

display(sample_submission.tail(2))
ImageId = []

for path in os.listdir(test_dir):

    # '456.jpg' to '456'

    path = path[:-4]

    path = int(path)

    # +1 because index starts at 1 in the submission file

    path = path + 1

    ImageId.append(path)
submission  = pd.DataFrame({

    "ImageId": ImageId,

    "Label": class_score

})

submission.sort_values(by=["ImageId"], inplace = True)

submission.to_csv("submission.csv", index=False)

submission[:10]
learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir="/tmp/models")
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5)
learn.save('stage1')
learn.load('stage1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(10, max_lr=slice(1e-6,1e-4))
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(6)