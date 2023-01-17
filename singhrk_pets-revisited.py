# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai.vision import *
bs = 64
path = untar_data(URLs.PETS)/'images'
path.ls()
tfms = get_transforms(max_rotate=25)
path/'images/boxer_27.jpg'
def get_ex(): return open_image(path/'boxer_27.jpg')



def plots_f(rows, cols, width, height, **kwargs):

    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

        rows,cols,figsize=(width,height))[1].flatten())]
plots_f(3,5,12,6,size=224)
plots_f(2,4,12,6,size=224, padding_mode='zeros')
plots_f(2,4,12,6,size=224, padding_mode='border')
plots_f(2,4,12,6,size=224, padding_mode='reflection')
tfms = get_transforms(max_rotate=20, max_warp=0.4, max_lighting=0.3, p_affine=1.)
plots_f(2,4,12,6,size=224)
tfms=get_transforms(max_rotate=20, max_zoom=1.3, max_lighting=0.3, max_warp=0.4, p_affine=1., p_lighting=1.)
src = ImageList.from_folder(path).split_by_rand_pct(0.2, seed=2)
def get_data(size, bs, padding_mode='reflection'):

    return (src.label_from_re(r'([^/]+)_\d+.jpg$')

           .transform(tfms, size=size, padding_mode=padding_mode)

           .databunch(bs=bs).normalize(imagenet_stats))
data = get_data(224, bs, 'zeros');data
def _plot(i,j,ax):

    x,y = data.train_ds[3]

    x.show(ax,y=y)

    

plot_multi(_plot, 3,3,figsize=(8,8))
data = get_data(224, bs);data
plot_multi(_plot, 3,3,figsize=(8,8))
n = gc.collect();n
learn = cnn_learner(data, models.resnet34, metrics=error_rate, bn_final=True)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, slice(5e-3), pct_start=0.8)
learn.recorder.plot_lr()
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, max_lr=slice(1e-5, 5e-3/5), pct_start=0.8)
learn.save('224_pct_08')
learn = cnn_learner(data, models.resnet34, metrics=error_rate, bn_final=True)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, slice(5e-3))
learn.recorder.plot_lr()
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-5, 1e-3))
learn.save('224_pct_03')
data = get_data(352, bs)

learn.data = data
learn.fit_one_cycle(2, max_lr=slice(1e-5, 5e-3/5))
learn.save('352_pct_03')