%matplotlib inline



import torch

import cv2

import matplotlib.pyplot as plt

import PIL

print(PIL.PILLOW_VERSION)



from fastai.vision import *

from fastai.metrics import error_rate

import fastai

fastai.__version__

import os

import pandas as pd



print(os.listdir('../input/'))
input_train_folder = '../input/train/'

print(os.listdir(input_train_folder)[:10])

input_test_folder = '../input/test/'

print(os.listdir(input_test_folder)[:10])
import shutil

train_folder = './data/train'

test_folder = './data/test'

#shutil.rmtree(train_folder)

if not os.path.exists(train_folder): 

    shutil.copytree(input_train_folder, train_folder)

if not os.path.exists(test_folder): 

    shutil.copytree(input_test_folder, test_folder)
tfms = get_transforms()

data = ImageDataBunch.from_folder(

    train_folder,

    test='../test',

    valid_pct = 0.2,

    bs = 32,

    size = 336,

    ds_tfms = tfms,

    num_workers = 0,

    ).normalize(imagenet_stats)

print(data.classes)

data.show_batch()
learn = create_cnn( data, models.resnet50, metrics=accuracy )
learn.fit_one_cycle(10)
# learn.unfreeze()

# learn.lr_find()

# learn.recorder.plot()
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
preds,y = learn.get_preds(ds_type=DatasetType.Test)

preds = np.argmax(preds, axis = 1)

preds_classes = [data.classes[i] for i in preds]
submission = pd.DataFrame({ 'file': os.listdir('../input/test'), 'species': preds_classes })

submission.to_csv('results.csv', index=False)
shutil.rmtree(train_folder)

shutil.rmtree(test_folder)
submission