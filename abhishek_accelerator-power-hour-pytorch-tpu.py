!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
!pip install wtfml==0.0.3

!pip install efficientnet_pytorch
import gc

import os

import torch

import albumentations



import numpy as np

import pandas as pd



import torch.nn as nn

from sklearn import metrics

from sklearn import model_selection

from torch.nn import functional as F



from wtfml.engine import Engine

from wtfml.utils import EarlyStopping

from wtfml.data_loaders.image import ClassificationDataLoader





import torch_xla.core.xla_model as xm

import torch_xla.distributed.parallel_loader as pl

import torch_xla.distributed.xla_multiprocessing as xmp



import efficientnet_pytorch
class EfficientNet(nn.Module):

    def __init__(self):

        super(EfficientNet, self).__init__()

        self.base_model = efficientnet_pytorch.EfficientNet.from_pretrained(

            'efficientnet-b0'

        )

        self.base_model._fc = nn.Linear(

            in_features=1280, 

            out_features=1, 

            bias=True

        )

        

    def forward(self, image, targets):

        out = self.base_model(image)

        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(out))

        return out, loss
# create folds

df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")

df["kfold"] = -1    

df = df.sample(frac=1).reset_index(drop=True)

y = df.target.values

kf = model_selection.StratifiedKFold(n_splits=5)



for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):

    df.loc[v_, 'kfold'] = f



df.to_csv("train_folds.csv", index=False)
# init model here

MX = EfficientNet()
def train():

    training_data_path = "../input/siic-isic-224x224-images/train/"

    df = pd.read_csv("/kaggle/working/train_folds.csv")

    device = xm.xla_device()

    epochs = 5

    train_bs = 32

    valid_bs = 16

    fold = 0



    df_train = df[df.kfold != fold].reset_index(drop=True)

    df_valid = df[df.kfold == fold].reset_index(drop=True)



    model = MX.to(device)



    mean = (0.485, 0.456, 0.406)

    std = (0.229, 0.224, 0.225)

    train_aug = albumentations.Compose(

        [

            albumentations.Normalize(

                mean, 

                std, 

                max_pixel_value=255.0, 

                always_apply=True

            ),

            albumentations.ShiftScaleRotate(

                shift_limit=0.0625, 

                scale_limit=0.1, 

                rotate_limit=15

            ),

            albumentations.Flip(p=0.5)

        ]

    )



    valid_aug = albumentations.Compose(

        [

            albumentations.Normalize(

                mean, 

                std, 

                max_pixel_value=255.0,

                always_apply=True

            )

        ]

    )



    train_images = df_train.image_name.values.tolist()

    train_images = [

        os.path.join(training_data_path, i + ".png") for i in train_images

    ]

    train_targets = df_train.target.values



    valid_images = df_valid.image_name.values.tolist()

    valid_images = [

        os.path.join(training_data_path, i + ".png") for i in valid_images

    ]

    valid_targets = df_valid.target.values



    train_loader = ClassificationDataLoader(

        image_paths=train_images,

        targets=train_targets,

        resize=None,

        augmentations=train_aug,

    ).fetch(

        batch_size=train_bs, 

        drop_last=True, 

        num_workers=4, 

        shuffle=True, 

        tpu=True

    )



    valid_loader = ClassificationDataLoader(

        image_paths=valid_images,

        targets=valid_targets,

        resize=None,

        augmentations=valid_aug,

    ).fetch(

        batch_size=valid_bs, 

        drop_last=False, 

        num_workers=2, 

        shuffle=False, 

        tpu=True

    )



    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        patience=3,

        threshold=0.001,

        mode="min"

    )



    es = EarlyStopping(patience=5, mode="min", tpu=True)

    eng = Engine(model, optimizer, device=device, use_tpu=True, tpu_print=25)



    for epoch in range(epochs):

        train_loss = eng.train(train_loader)

        valid_loss = eng.evaluate(valid_loader)

        xm.master_print(f"Epoch = {epoch}, LOSS = {valid_loss}")

        scheduler.step(valid_loss)



        es(valid_loss, model, model_path=f"model_fold_{fold}.bin")

        if es.early_stop:

            xm.master_print("Early stopping")

            break

        gc.collect()
def _mp_fn(rank, flags):

    torch.set_default_tensor_type('torch.FloatTensor')

    a = train()
FLAGS={}

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')