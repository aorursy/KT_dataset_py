!pip3 install fastai==1.0.42
#!pip3 install git+https://github.com/fastai/fastai.git

#!pip3 install git+https://github.com/pytorch/pytorch
import torch

print(torch.__version__)
from fastai.vision import * 

from fastai import *
#import fastai; 

#fastai.show_install(1)
planet = untar_data(URLs.PLANET_TINY)

planet_tfms = get_transforms(flip_vert=False, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
data = (ImageItemList.from_csv(planet, 'labels.csv', folder='train', suffix='.jpg')

        .random_split_by_pct()

        .label_from_df(label_delim=' ')

        .transform(planet_tfms, size=128)

        .databunch()

        .normalize(imagenet_stats))
data.show_batch(rows=2, figsize=(9,7))
#??create_cnn()
#learn = create_cnn(data, models.resnet18, metrics=Fbeta(beta=2))

#learn.fit(5)
learn = create_cnn(data, models.resnet18)

learn.fit_one_cycle(5,1e-2)

learn.save('mini_train')
learn.show_results(rows=3, figsize=(12,15))
#from fastai.metrics import FBeta
learn = create_cnn(data, models.resnet50, metrics=[accuracy_thresh])
learn.lr_find()
learn.recorder.plot()

lr = 3e-2
#learn.fit_one_cycle(5, slice(lr))
learn.fit_one_cycle(5,1e-2)

learn.save('mini_train')
learn.show_results(rows=3, figsize=(24,30))
