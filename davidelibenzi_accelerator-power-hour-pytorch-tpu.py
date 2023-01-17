!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
!pip install git+https://github.com/abhishekkrthakur/wtfml
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

from wtfml.utils import EarlyStopping
from wtfml.data_loaders.image import ClassificationDataLoader


import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

import efficientnet_pytorch
class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.base_model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
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
import torch
from tqdm import tqdm
from wtfml.utils import AverageMeter

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    _xla_available = True
except ImportError:
    _xla_available = False

try:
    from apex import amp

    _apex_available = True
except ImportError:
    _apex_available = False
    


def reduce_fn(vals):
    return sum(vals) / len(vals)


class Engine:
    @staticmethod
    def train(
        data_loader,
        model,
        optimizer,
        device,
        scheduler=None,
        accumulation_steps=1,
        use_tpu=False,
        fp16=False,
        bs=1
    ):
        if use_tpu and not _xla_available:
            raise Exception(
                "You want to use TPUs but you dont have pytorch_xla installed"
            )
        if fp16 and not _apex_available:
            raise Exception("You want to use fp16 but you dont have apex installed")
        if fp16 and use_tpu:
            raise Exception("Apex fp16 is not available when using TPUs")
        if fp16:
            accumulation_steps = 1
        losses = AverageMeter()
        predictions = []
        model.train()
        data_loader = pl.ParallelLoader(data_loader, [device]).per_device_loader(device)
 
        for b_idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            _, loss = model(**data)

            loss.backward()
            xm.optimizer_step(optimizer)
            if scheduler is not None:
                scheduler.step()
            reduced_loss = xm.mesh_reduce('loss_reduce', loss, reduce_fn)
            losses.update(reduced_loss.item(), bs)

        return losses.avg

    @staticmethod
    def evaluate(data_loader, model, device, use_tpu=False, bs=1):
        losses = AverageMeter()
        final_predictions = []
        final_targets = []
        model.eval()
        with torch.no_grad():
            data_loader = pl.ParallelLoader(data_loader, [device]).per_device_loader(device)
            for b_idx, data in enumerate(data_loader):
                _, loss = model(**data)
                reduced_loss = xm.mesh_reduce('loss_reduce', loss, reduce_fn)
                losses.update(reduced_loss.item(), bs)

        return losses.avg
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
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
            albumentations.Flip(p=0.5)
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".png") for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".png") for i in valid_images]
    valid_targets = df_valid.target.values

    train_loader = ClassificationDataLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug,
    ).fetch(
        batch_size=train_bs, 
        drop_last=True, 
        num_workers=0, 
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
        num_workers=0, 
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

    es = EarlyStopping(patience=5, mode="min")

    for epoch in range(epochs):
        train_loss = Engine.train(
            train_loader, 
            model, 
            optimizer, 
            device=device, 
            use_tpu=True,
            bs=train_bs)
        
        valid_loss = Engine.evaluate(
            valid_loader, 
            model, 
            device=device, 
            use_tpu=True,
            bs=valid_bs
        )
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
