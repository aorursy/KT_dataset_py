!pip install pretrainedmodels
from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



import torch

import fastai

from fastai import vision



import pretrainedmodels as pm
%matplotlib inline
import glob

import os

import shutil

from typing import List



import numpy as np

import pandas as pd

from sklearn import model_selection



PREFIX = "./data"

SEED = 42

TEST_SIZE = 0.2



def _filepaths_to_dataframe(paths: List[str]) -> pd.DataFrame:

    """Converts filepaths to a Pandas DataFrame."""

    results = {"label": [], "filename": []}

    for path in paths:

        _, _, _, _, _label, _ = path.split('/')

        results["label"].append(_label)

        results["filename"].append(path)

    df = (pd.DataFrame

            .from_dict(results))

    return df





def _make_interim_training_data(prefix: str, df: pd.DataFrame) -> None:

    if not os.path.isdir(f"{prefix}/interim/train"):

        os.makedirs(f"{prefix}/interim/train")



    for _, row in df.iterrows():

        label, path = row

        filename = (os.path

                      .basename(path))

        if not os.path.isdir(f"{prefix}/interim/train/{label}"):

            os.mkdir(f"{prefix}/interim/train/{label}")

        shutil.copy(path, f"{prefix}/interim/train/{label}/{filename}")



        

def _make_interim_validation_data(prefix: str, df: pd.DataFrame) -> None:

    if not os.path.isdir(f"{prefix}/interim/valid"):

        os.makedirs(f"{prefix}/interim/valid")



    for _, row in df.iterrows():

        label, path = row

        filename = (os.path

                      .basename(path))

        if not os.path.isdir(f"{prefix}/interim/valid/{label}"):

            os.mkdir(f"{prefix}/interim/valid/{label}")

        shutil.copy(path, f"{prefix}/interim/valid/{label}/{filename}")



        

filepaths = glob.glob(f"../input/train/train/*/*.jpg", recursive=True)

df = _filepaths_to_dataframe(filepaths)

prng = np.random.RandomState(SEED)



training_df, validation_df = model_selection.train_test_split(df,

                                                              test_size=TEST_SIZE,

                                                              random_state=prng,

                                                              stratify=df["label"])

    

if not os.path.isdir(PREFIX):

    os.mkdir(PREFIX)

_make_interim_training_data(PREFIX, training_df)

_make_interim_validation_data(PREFIX, validation_df)





fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 8))



_ = (df.loc[:, "label"]

       .value_counts()

       .plot

       .bar(ax=axes[0], title="Raw"))



_ = (training_df.loc[:, "label"]

                .value_counts()

                .plot

                .bar(ax=axes[1], title="Training"))



_ = (validation_df.loc[:, "label"]

                  .value_counts()

                  .plot

                  .bar(ax=axes[2], title="Validation"))
def set_seed(seed):

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True



set_seed(42)
_transform_kwargs = {"do_flip": True,

                     "flip_vert": True,  # default is False

                     "max_rotate": 180,  # default is 10

                     "max_zoom": 1.2,    # default is 1.1

                     "max_lighting": 0.2,

                     "max_warp": 0.2,

                     "p_affine": 0.75,

                     "p_lighting": 0.7,

                    }

        

_transforms = vision.get_transforms(**_transform_kwargs)



_data_bunch_kwargs = {"path": "./data/interim",

                      "train": "train",

                      "valid": "valid",

                      "bs": 16,

                      "size": 448,

                      "ds_tfms": _transforms,

                      "test": "../../../input/test/test",  ## hack to access the test data without copying to ./data

                     }



image_data_bunch = (vision.ImageDataBunch

                          .from_folder(**_data_bunch_kwargs)

                          .normalize())
image_data_bunch.train_ds
image_data_bunch.valid_ds
image_data_bunch.test_ds
image_data_bunch.show_batch(figsize=(20,20))
_base_arch = lambda arg: pm.se_resnext101_32x4d(num_classes=1000, pretrained="imagenet")

learner = vision.cnn_learner(image_data_bunch,

                             base_arch=_base_arch,

                             pretrained=True,

                             metrics=vision.error_rate,

                             model_dir="/kaggle/working/models/se-resnext101-32x4d")
learner.summary()
learner.lr_find()
(learner.recorder

        .plot())
def find_optimal_lr(recorder):

    """Extract the optimal learning rate from recorder data."""

    optimal_lr = 0

    minimum_loss = float("inf")

    for loss, lr in zip(recorder.losses, recorder.lrs):

        if loss < minimum_loss:

            optimal_lr = lr

            minimum_loss = loss

    return optimal_lr, minimum_loss

# define a callback that stores state of "best" model.

# N.B. best model is re-loaded when training completes

_save_model_kwargs = {"every": "improvement",

                      "monitor": "valid_loss",

                      "name": "best-model-stage-1"}

_save_model = (fastai.callbacks

                     .SaveModelCallback(learner, **_save_model_kwargs))



# if validation loss < training loss either learning rate too low or not enough training epoch

learner.fit_one_cycle(15, callbacks=[_save_model])
clf_interp = (vision.ClassificationInterpretation

                    .from_learner(learner))
clf_interp.plot_top_losses(16, figsize=(20,20))
clf_interp.plot_confusion_matrix()
clf_interp.most_confused()
learner.unfreeze()
learner.summary()
learner.lr_find()
(learner.recorder

        .plot())
_save_model_kwargs = {"every": "improvement",

                      "monitor": "valid_loss",

                      "name": "best-model-stage-2"}

_save_model = (fastai.callbacks

                     .SaveModelCallback(learner, **_save_model_kwargs))

learner.fit_one_cycle(15, max_lr=slice(1e-6, 1e-4), callbacks=[_save_model])
clf_interp = (vision.ClassificationInterpretation

                    .from_learner(learner))
clf_interp.plot_confusion_matrix()
predicted_class_probabilities, _ = learner.TTA(ds_type=fastai.basic_data.DatasetType.Test)
_predicted_classes = (predicted_class_probabilities.argmax(dim=1)

                                                   .numpy())

_class_labels = np.array(['cbb','cbsd','cgm','cmd','healthy'])

_predicted_class_labels = _class_labels[_predicted_classes]



_filenames = np.array([item.name for item in image_data_bunch.test_ds.items])



submission = (pd.DataFrame

                .from_dict({'Category': _predicted_class_labels,'Id': _filenames}))
submission.head()
submission.to_csv('submission.csv', header=True, index=False)
shutil.rmtree(PREFIX)  # necessary not to overwhlem Kaggle with unused output files