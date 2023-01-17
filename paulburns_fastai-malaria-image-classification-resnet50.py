# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from fastai import *

from fastai.vision import *

from fastai.vision.gan import *

from fastai.callbacks.hooks import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



print(os.listdir("../input/cell_images/cell_images"))



# Any results you write to the current directory are saved as output.
img_dir='../input/cell_images/cell_images/'

path = Path(img_dir)
data = ImageDataBunch.from_folder(path, 

                                  train=".",

                                  valid_pct=0.2, 

                                  ds_tfms=get_transforms(flip_vert=True, max_warp=0),

                                  size=224,bs=64, 

                                  num_workers=0).normalize(imagenet_stats)
data.show_batch(rows=5)
print(f'Classes: \n {data.classes}')
learn = cnn_learner(data, models.resnet50, metrics=[accuracy, error_rate], model_dir="/temp/model/")
learn.lr_find()

learn.recorder.plot()
learn_rate = 1e-01
learn.fit_one_cycle(5, slice(learn_rate))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
learn.save('stage-1-rn50')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-5, learn_rate/5))
learn.save('stage-2-rn50')
learn.unfreeze()

learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
from fastai.callbacks.hooks import *



idx=0

x,y = data.valid_ds[idx]

eval_model = learn.model.eval();

xb,_ = data.one_item(x)

xb_im = Image(data.denorm(xb)[0])

xb = xb.cuda()
def hooked_backward(cat=y):

    with hook_output(eval_model[0]) as hook_a: 

        with hook_output(eval_model[0], grad=True) as hook_g:

            preds = eval_model(xb)

            preds[0,int(cat)].backward()

    return hook_a, hook_g
def show_heatmap(hm):

    _,ax = plt.subplots()

    xb_im.show(ax)

    ax.imshow(hm, alpha=0.6, extent=(0,352,352,0),

              interpolation='bilinear', cmap='magma');
hook_a, hook_g = hooked_backward()
acts  = hook_a.stored[0].cpu()

acts.shape
grad = hook_g.stored[0][0].cpu()

grad_chan = grad.mean(1).mean(1)

grad.shape,grad_chan.shape
mult = (acts*grad_chan[...,None,None]).mean(0)

show_heatmap(mult)