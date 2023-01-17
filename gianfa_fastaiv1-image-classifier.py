# %reload_ext autoreload

# %autoreload 2

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
os.listdir('../input/')
train_folder = '../input/train/train/'

test_folder = '../input/test1/test1/'

print(os.listdir(train_folder)[:10])

print(os.listdir(test_folder)[:10])
np.random.seed(2)

fnames = get_image_files(train_folder)

print(fnames[:5])

pat = re.compile(r'(cat|dog)\.\d+\.jpg') # we specify a regex for finding cat or dog images
sz = 64

bs = 64

data = ImageDataBunch.from_name_re(

                                train_folder,

                                fnames,

                                pat,

                                ds_tfms=get_transforms(),

                                size=sz, bs=bs,

                                valid_pct = 0.25,

                                num_workers = 0, # for code safety on kaggle

).normalize(imagenet_stats)

data
data.classes
data.show_batch(rows=4, figsize=(7,6))
learn = cnn_learner(

    data,

    models.resnet34,

    metrics=error_rate,

    model_dir="/tmp/model/"

)
learn.data
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, max_lr=slice(2e-3))
learn.unfreeze()

learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

print(len(data.valid_ds)==len(losses)==len(idxs))
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(5,5), dpi=100)
mc = interp.most_confused(min_val=2)

mcc = [x[0] for x in mc[:5]]

mcc
train = pd.DataFrame(os.listdir(train_folder))
a = ['0']

train.sample(n=10, random_state=1)
item1_path = data.items[100]

print(item1_path)

item1 = data.open( item1_path )

item1
pred_class, pred_idx, outputs = learn.predict(item1)

probs = torch.nn.functional.softmax(np.log(outputs), dim=0)

print(pred_class)

print(probs)
data_test = ImageList.from_folder(test_folder).split_none().label_empty()

data_test
dst = data_test.train.x[:20]

dst
learn.data
data.add_test(items=dst)

data
learn.data = data

learn.data
pred_probs, pred_class = learn.get_preds(ds_type=DatasetType.Test)
print(pred_probs)

print(pred_class)

print((pred_probs.numpy()[:,0]>0.5)+0)
df = pd.DataFrame(os.listdir(test_folder))

print(len(df))

df.head()
img_idx = 8

print(pred_class[img_idx])

data_test.train.x[img_idx]

plt.imshow(plt.imread(test_folder+df[0].iloc[img_idx]))

# test_folder+df[0]


submission_data = [ids, pred_class]



df = pd.DataFrame(submission_data).T

df.columns = ['id','label']

df.head()
df.to_csv('./kaggle_catsdogs.csv', index=False)

print( os.path.exists('./kaggle_catsdogs.csv') )
!rm -rf dogscats