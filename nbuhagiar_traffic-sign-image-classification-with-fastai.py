# General Data Science

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# Machine Learning

from fastai.vision import ImageDataBunch, get_transforms, cnn_learner

from fastai.vision.data import imagenet_stats

from fastai.vision.models import resnet34

from fastai.vision.learner import ClassificationInterpretation

from fastai.metrics import error_rate



# Miscellaneous

import os

from pathlib import Path

from PIL import Image

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
data_path = Path("/kaggle/input/gtsrb-german-traffic-sign/")

data = ImageDataBunch.from_folder(data_path/"train",

                                  ds_tfms=get_transforms(do_flip=False),

                                  size=224,

                                  bs=64,

                                  valid_pct=0.1,

                                  seed=0).normalize(imagenet_stats)

data
data.show_batch(rows=3, figsize=(7,6))
data.c
learn = cnn_learner(data, resnet34, metrics=error_rate, model_dir=Path("/kaggle/working/model"))

learn.model
learn.fit_one_cycle(4)
interpretation = ClassificationInterpretation.from_learner(learn)

interpretation.plot_top_losses(9)
interpretation.most_confused(min_val=2)
def img_comparison(label_0, label_1):

    """

    Plot samples of two image classes side-by-side.

    

    Args:

        label_0: int, first class label of interest

        label_1: int, second class label of interest

    """

    

    fig, ax = plt.subplots(1, 2)

    image_0 = np.array(Image.open(f"/kaggle/input/gtsrb-german-traffic-sign/meta/{label_0}.png"))

    image_1 = np.array(Image.open(f"/kaggle/input/gtsrb-german-traffic-sign/meta/{label_1}.png"))

    ax[0].imshow(image_0)

    ax[1].imshow(image_1)

    fig.tight_layout()
img_comparison(2, 5)
img_comparison(7, 8)
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-4))