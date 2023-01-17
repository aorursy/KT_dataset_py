!pip install pretrainedmodels
from PIL import Image

import shutil



import numpy as np

import matplotlib.pyplot as plot

import pandas as pd



import torch

import fastai

from fastai import vision



import pretrainedmodels as pm
%matplotlib inline
def set_seed(seed):

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True



set_seed(42)
transform_kwargs = {"do_flip": True,

                    "flip_vert": True,

                    "max_rotate": 180,

                    "max_zoom": 1.1,

                    "max_lighting": 0.2,

                    "max_warp": 0.2,

                    "p_affine": 0.75,

                    "p_lighting": 0.7}

        

transforms = vision.get_transforms(**transform_kwargs)



data_bunch_kwargs = {"path": "../input/cassava-disease/train",

                     "train": "train",

                     "valid_pct": 0.1,

                     "size": 448,

                     "bs": 16,

                     "ds_tfms": transforms,

                     "test": "../extraimages/extraimages"}



image_data_bunch = (vision.ImageDataBunch

                          .from_folder(**data_bunch_kwargs)

                          .normalize())
shutil.copytree("../input/cassava-disease-classification/models/", "./models")
_base_arch = lambda arg: pm.se_resnext101_32x4d(num_classes=5, pretrained=None)

learner = vision.cnn_learner(image_data_bunch, base_arch=_base_arch, pretrained=False, metrics=vision.error_rate, model_dir="/kaggle/working/models/se-resnext101-32x4d")

_ = learner.load("best-model-stage-2")
predicted_probabilities, _ = learner.TTA(ds_type=fastai.basic_data.DatasetType.Test)
predicted_class_probabilities, _predicted_classes = predicted_probabilities.max(dim=1)

class_labels = np.array(['cbb','cbsd','cgm','cmd','healthy'])

predicted_class_labels = class_labels[_predicted_classes]

shutil.copytree("../input/cassava-disease/train/train/", "./data/train")

shutil.copytree("../input/cassava-disease/test/test/", "./data/test")



threshold = 0.95  # only include pseudo-labeled images where model is sufficiently confident in its prediction

filenames = [item.name for item in learner.data.test_ds.items]

for predicted_class_label, predicted_class_probability, filename in zip(predicted_class_labels, predicted_class_probabilities, filenames):

    if predicted_class_probability > threshold:

        shutil.copy(f"../input/cassava-disease/extraimages/extraimages/{filename}", f"./data/train/{predicted_class_label}/{filename}")
transform_kwargs = {"do_flip": True,

                    "flip_vert": True,

                    "max_rotate": 180,

                    "max_zoom": 1.1,

                    "max_lighting": 0.2,

                    "max_warp": 0.2,

                    "p_affine": 0.75,

                    "p_lighting": 0.7}

        

transforms = vision.get_transforms(**transform_kwargs)



data_bunch_kwargs = {"path": "./data/train",

                     "train": "train",

                     "valid_pct": 0.1,

                     "size": 448,

                     "bs": 16,

                     "ds_tfms": transforms,

                     "test": "../test"}



image_data_bunch = (vision.ImageDataBunch

                          .from_folder(**data_bunch_kwargs)

                          .normalize())
_base_arch = lambda arg: pm.se_resnext101_32x4d(num_classes=5, pretrained=None)

learner = vision.cnn_learner(image_data_bunch, base_arch=_base_arch, pretrained=False, metrics=vision.error_rate, model_dir="/kaggle/working/models/se-resnext101-32x4d")

_ = learner.load("best-model-stage-2")
learner.lr_find()
(learner.recorder

        .plot())
_save_model_kwargs = {"every": "improvement",

                      "monitor": "valid_loss",

                      "name": "best-model-stage-3"}

_save_model = (fastai.callbacks

                     .SaveModelCallback(learner, **_save_model_kwargs))

learner.fit_one_cycle(15, max_lr=slice(None, 1e-6, None), callbacks=[_save_model])
predicted_probabilities, _ = learner.TTA(ds_type=fastai.basic_data.DatasetType.Test)
_, _predicted_classes = predicted_probabilities.max(dim=1)

_predicted_class_labels = class_labels[_predicted_classes]



_filenames = np.array([item.name for item in image_data_bunch.test_ds.items])



submission = (pd.DataFrame

                .from_dict({'Category': _predicted_class_labels,'Id': _filenames}))
submission.to_csv('submission-using-pseudo-labels.csv', header=True, index=False)
shutil.rmtree("./data") # remove unnecessary output files!