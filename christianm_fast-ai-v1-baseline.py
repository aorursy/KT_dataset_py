%reload_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from fastai import *

from fastai.vision import *

from fastai.callbacks.hooks import *



import os

print(os.listdir("../input/cell_images/cell_images/"))
path=Path("../input/cell_images/cell_images/")

path
help(get_transforms)
tfms = get_transforms(

    flip_vert=True, 

    max_lighting=0.1, 

    max_zoom=1.1, 

    max_warp=0.1,

    p_affine=0.75,

    p_lighting=0.75

)
src = ImageList.from_folder(path).split_by_rand_pct().label_from_folder()

src
def get_data(src, size, bs, tfms):

    return (src.transform(tfms, size=size)

        .databunch(bs=bs)

        .normalize())
data = get_data(src, 64, 128, tfms)
data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.resnet18, metrics=[accuracy], callback_fns=[ShowGraph], model_dir="/tmp/model/")
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5,1e-3)
learn.save('stage-1')



learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.unfreeze()

learn.fit_one_cycle(5,1e-5)
learn.data = get_data(src, 256, 128, tfms)



learn.freeze()

learn.fit_one_cycle(5,1e-3)
learn.unfreeze()

learn.fit_one_cycle(5,1e-5)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)