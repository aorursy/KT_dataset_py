!pip show fastai
import numpy as np 

import pandas as pd 

from fastai.vision import *

from fastai.vision.models import *



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sn

import pickle

import csv

import glob



from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

from PIL import Image



import shutil, os
# SETUP

MODEL=resnet50

BATCH = 64

SIZE = 320



DATA_PATH = '../input/'

PATH = "../working/car/"
os.makedirs(PATH,exist_ok=False)

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
def accuracy(preds, targs):

    preds = torch.max(preds, dim=1)[1]

    return (preds==targs).float().mean()
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/sample.csv")

train_df.head()
train_df.info()
train_df.Category.value_counts()
# Пример картинок

fig, axs = plt.subplots(1,4,figsize=(15,10))

i = 0

for ax in axs:

    path_img = DATA_PATH+'train/train/'+str(i)

    fnames = get_image_files(path_img)

    im = open_image(fnames[0])

    im.show(ax=ax, title=f'{i}')

    i+=1
image = Image.open(DATA_PATH+'/train/train/0/100380.jpg')

imgplot = plt.imshow(image)

plt.show()

image.size
#tfms = None

tfms = get_transforms(max_zoom=1., max_warp=0.2, max_lighting=0.3,

                     xtra_tfms=[cutout(n_holes=(1,20))]

                     )
data = ImageDataBunch.from_folder(DATA_PATH, train=PATH+'train/train/', test='test/test_upload/',

                                  ds_tfms=tfms, padding_mode='zeros',

                                  valid_pct=0.1, size=SIZE, 

                                  classes=['0','1','2','3','4','5','6','7','8','9'],

                                  bs=BATCH, num_workers=0).normalize(imagenet_stats)

data.path = pathlib.Path(PATH)  ## IMPORTANT for PyTORCH to create tmp directory which won't be otherwise allowed on Kaggle Kernel directory structure
data
data.show_batch(rows=4, figsize=(12,9))
learn = cnn_learner(data, MODEL, metrics=accuracy, model_dir=PATH)
learn.lr_find()

learn.recorder.plot()
learn.recorder.plot_losses()

learn.recorder.plot_metrics()

learn.recorder.plot()
# Test Valid

accuracy(*learn.get_preds())
learn.save('stage-1-resnet50')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15,11))
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-7,1e-5),)

learn.recorder.plot()

learn.recorder.plot_losses()

learn.recorder.plot_metrics()
# Test Valid

accuracy(*learn.get_preds())
accuracy(*learn.TTA())
learn.freeze()
learn.save('stage-2-resnet50', return_path=True)
learn = cnn_learner(data, MODEL, metrics=accuracy, model_dir=PATH)

learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(30, max_lr=1e-4)

learn.recorder.plot()

learn.recorder.plot_losses()

learn.recorder.plot_metrics()
# Test Valid

accuracy(*learn.get_preds())
accuracy(*learn.TTA())
learn.freeze()

learn.save('stage-3-resnet50', return_path=True)
learn.export()
submission = pd.read_csv('../input/sample.csv')

submission.shape
submission.head()
data.train_ds.x
data.test_ds.x
preds_test, _ = learn.get_preds(DatasetType.Test)
preds_test[:10][0]
preds_test_s = pd.DataFrame(preds_test)

preds_test_s.to_csv('preds_sub.csv', index=False)

preds_test_s.shape
preds_test = np.argmax(preds_test, axis=1)

preds_test = preds_test.numpy()
preds_test.shape
preds_test[:10]
fnames = [f.name for f in learn.data.test_ds.items]

submission = pd.DataFrame({'Id':fnames, 'Category':preds_test}, columns=['Id', 'Category'])
submission.to_csv('submission_v23.csv', index=False)
submission.head()
preds_test_tta, _ = learn.TTA(ds_type=DatasetType.Test)

#preds_test_tta_2, = learn.TTA(ds_type=DtasetType.Test)[0]
preds_test_tta_s = pd.DataFrame(preds_test_tta)

preds_test_tta_s.to_csv('preds_tta_sub.csv', index=False)

preds_test_tta_s.shape
preds_test_tta = np.argmax(preds_test_tta, axis=1)

preds_test_tta = preds_test_tta.numpy()
print(preds_test_tta[:15])
fnames = [f.name for f in learn.data.test_ds.items]

submission = pd.DataFrame({'Id':fnames, 'Category':preds_test_tta}, columns=['Id', 'Category'])
submission['Category'] = preds_test_tta

submission.to_csv('tta_submission_v23.csv', index=False)
submission.head()
print(os.listdir(PATH))
shutil.copy2( PATH+'export.pkl', '../working/export.pkl')
shutil.rmtree(PATH)
print(os.listdir('../working/'))