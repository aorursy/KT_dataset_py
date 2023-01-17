%reload_ext autoreload

%autoreload 2

%matplotlib inline
import os

print(os.listdir("../input/v2-plant-seedlings-dataset/"))
from fastai import *

from fastai.vision import *
PATH = "../input/v2-plant-seedlings-dataset/nonsegmentedv2/"
sz=128

bs=64
tfms = get_transforms()

data = ImageDataBunch.from_folder(PATH, valid_pct=0.2,

    ds_tfms=tfms, size=sz,bs=bs, num_workers=0).normalize(imagenet_stats)
print(f'We have {len(data.classes)} different types of seedlings\n')

print(f'Types: \n {data.classes}')
data.show_batch(8, figsize=(20,15))
from os.path import expanduser, join, exists

from os import makedirs

cache_dir = expanduser(join('~', '.torch'))

if not exists(cache_dir):

    makedirs(cache_dir)

models_dir = join(cache_dir, 'models')

if not exists(models_dir):

    makedirs(models_dir)



# copy time!

!cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth
learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir='/output/model/',callback_fns=ShowGraph)
lrf=learn.lr_find()

learn.recorder.plot()
lr=1e-2
learn.fit_one_cycle(2,lr)
learn.save('seedlings-stage-1')
learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(5e-6, 5e-4))
learn.save('seedlings-stage-2')
interp = ClassificationInterpretation.from_learner(learn,tta=True)
interp.plot_top_losses(16, figsize=(20,14))
interp.plot_confusion_matrix(figsize=(12,12), dpi=100, normalize=True, norm_dec=2, cmap=plt.cm.YlGn)