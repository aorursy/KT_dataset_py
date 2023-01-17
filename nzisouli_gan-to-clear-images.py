# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from pathlib import Path

import fastai

from fastai.vision import *

from fastai.callbacks import *

from fastai.vision.gan import *

from distutils.dir_util import copy_tree
input_path = Path("/kaggle/input/dance-images")

path_cl= input_path/"clear"

path_bl= input_path/"blurry"
path = Path("/tmp/model")

model_path_cl = path/"clear"

model_path_bl = path/"blurry"

model_path_cl.mkdir(parents=True, exist_ok=True)

model_path_bl.mkdir(parents=True, exist_ok=True)

copy_tree(str(path_cl), str(path/"clear"))

copy_tree(str(path_bl), str(path/"blurry"))
bs,size=4, 256

arch = models.resnet34

src = ImageImageList.from_folder(model_path_bl).split_by_rand_pct(0.1, seed=42)
def get_data(bs,size):

    data = (src.label_from_func(lambda x: model_path_cl/x.name)

           .transform(get_transforms(max_zoom=0), size=size, tfm_y=True)

           .databunch(bs=bs, num_workers=0).normalize(imagenet_stats, do_y=True))



    #data.c = 3

    return data
data_gen = get_data(bs,size)

data_gen.show_batch()
y_range = (-3.,3.)

loss_gen = MSELossFlat()
learn_gen = unet_learner(data_gen, arch, blur=True, norm_type=NormType.Weight,

                         self_attention=True, y_range=y_range, loss_func=loss_gen)
learn_gen.lr_find()

learn_gen.recorder.plot()
lr = 1e-2

learn_gen.fit_one_cycle(4)
learn_gen.unfreeze()

learn_gen.fit_one_cycle(5, slice(1e-4,lr))

learn_gen.show_results()
learn_gen.save('gen-pre')

torch.cuda.empty_cache()
learn_gen.load('gen-pre');

name_gen = 'image_gen'

path_gen = path/name_gen

path_gen.mkdir(exist_ok=True)
def save_preds(dl):

    i=0

    names = dl.dataset.items

    

    for b in dl:

        preds = learn_gen.pred_batch(batch=b, reconstruct=True)

        for o in preds:

            o.save(path_gen/names[i].name)

            i += 1
save_preds(data_gen.fix_dl)
learn_gen=None

torch.cuda.empty_cache()

gc.collect()
def get_crit_data(classes, bs, size):

    src = ImageList.from_folder(path, include=classes).split_by_rand_pct(0.1, seed=42)

    ll = src.label_from_folder(classes=classes)

    data = (ll.transform(get_transforms(max_zoom=2.), size=size)

           .databunch(bs=bs).normalize(imagenet_stats))

    #data.c = 3

    return data
data_crit = get_crit_data([name_gen, 'clear'], bs=bs, size=size)

data_crit.show_batch()
loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())

learn_critic = Learner(data_crit, gan_critic(), metrics=accuracy_thresh_expand, loss_func=loss_critic)
learn_critic.lr_find()

learn_critic.recorder.plot()
lr = 1e-4

learn_critic.fit_one_cycle(10, lr)
learn_critic.save('critic-pre')
learn_crit=None

torch.cuda.empty_cache()

gc.collect()
data_crit = get_crit_data(['blurry', 'clear'], bs=bs, size=size)

learn_crit = Learner(data_crit, gan_critic(), metrics=None, loss_func=loss_critic).load('critic-pre')

learn_gen = unet_learner(data_gen, arch, blur=True, norm_type=NormType.Weight,

                         self_attention=True, y_range=y_range, loss_func=loss_gen).load('gen-pre')
switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)

learn = GANLearner.from_learners(learn_gen, learn_crit, weights_gen=(1.,50.), show_img=True, switcher=switcher,

                                 opt_func=partial(optim.Adam, betas=(0.,0.99)))

learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))
learn.lr_find()

learn.recorder.plot()
lr = 1e-4

learn.fit(30, lr)
bs,size = 1, 512

data = get_data(bs, size)

learn.data = data

gc.collect()
learn.fit(10, lr/2)
learn.show_results()
learn.save('gan')
m = learn.model.eval();
fn = "/kaggle/input/non-artificial-blurred/DSCN0140.JPG"

x = open_image(fn);

xb,_ = data.one_item(x)

xb_im = Image(data.denorm(xb)[0])

xb = xb.cuda()

xb_im
pred = m(xb)

pred_im = Image(data.denorm(pred.detach())[0])

pred_im