

!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
# this script transports l5kit and dependencies

!pip -q install pymap3d==2.1.0 

!pip -q install protobuf==3.12.2 

!pip -q install transforms3d 

!pip -q install zarr 

!pip -q install ptable

 

!pip -q install --no-dependencies l5kit


import numpy as np

import torch

import gc, os



import warnings

warnings.filterwarnings("ignore")

import numpy as np

import os

import torch

from multiprocessing import Pool

import random

import bz2

import pickle

from torch.nn import functional as f





from torch import nn, optim

from torch.utils.data import DataLoader,Dataset

from torchvision.models.resnet import resnet18

from tqdm import tqdm

from typing import Dict,Tuple

from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,ExponentialLR





from l5kit.evaluation import write_pred_csv

from l5kit.data import LocalDataManager

from l5kit.data import LocalDataManager,filter_agents_by_labels,get_combined_scenes

from l5kit.data import ChunkedDataset

from l5kit.dataset import EgoDataset,AgentDataset

from l5kit.rasterization import build_rasterizer





from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset

from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS

from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace,rmse,average_displacement_error_mean







from pathlib import Path

from tempfile import gettempdir

import pandas as pd







!pip install pytorch-lightning


import pytorch_lightning as pl

from pytorch_lightning.loggers import CSVLogger


DIR_INPUT = '../input/lyft-motion-prediction-autonomous-vehicles/' #data files



SINGLE_MODE_SUBMISSION = f"{DIR_INPUT}/single_mode_sample_submission.csv"

MULTI_MODE_SUBMISSION = f"{DIR_INPUT}/multi_mode_sample_submission.csv"



DEBUG = False

VALIDATION = False











cfg = {

    'format_version': 4,

    'model_params': {

        'model_architecture': 'resnet50',

        'history_num_frames': 10,

        'history_step_size': 1,

        'history_delta_time': 0.1,

        'future_num_frames': 50,

        'future_step_size': 1,

        'future_delta_time': 0.1

    },

    

    'raster_params': {

        'raster_size': [350, 350],

        'pixel_size': [0.5, 0.5],

        'ego_center': [0.25, 0.5],

        'map_type': 'py_semantic',

        'satellite_map_key': 'aerial_map/aerial_map.png',

        'semantic_map_key': 'semantic_map/semantic_map.pb',

        'dataset_meta_key': 'meta.json',

        'filter_agents_threshold': 0.5

    },

    

    'train_data_loader': {

        'key': 'scenes/train.zarr',

        'batch_size': 8,

        'shuffle': True,

        'num_workers': 4

    },

    

     'valid_data_loader': {

        'key': 'scenes/validate.zarr',

        'batch_size': 8,

        'shuffle': False,

        'num_workers': 4

    },

    

    'sample_data_loader': {

        'key': 'scenes/sample.zarr',

        'batch_size': 16,

        'shuffle': False,

        'num_workers': 0

    },

    

    'train_params': {

        'max_num_steps': 2334 if DEBUG else 20000,

        'checkpoint_every_n_steps': 2000,

        

        # 'eval_every_n_steps': -1

    }

}





# set env variable for data

os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT

dm = LocalDataManager(None)









# --- Function utils ---

# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py

import numpy as np



import torch

from torch import Tensor





def pytorch_neg_multi_log_likelihood_batch(

    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor

) -> Tensor:

    """

    Compute a negative log-likelihood for the multi-modal scenario.

    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:

    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    https://leimao.github.io/blog/LogSumExp/

    Args:

        gt (Tensor): array of shape (bs)x(time)x(2D coords)

        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)

        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample

        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep

    Returns:

        Tensor: negative log-likelihood for this example, a single float number

    """

    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"

    batch_size, num_modes, future_len, num_coords = pred.shape



    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"

    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"

    #print(confidences)

    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"

    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"

    # assert all data are valid

    assert torch.isfinite(pred).all(), "invalid value found in pred"

    assert torch.isfinite(gt).all(), "invalid value found in gt"

    assert torch.isfinite(confidences).all(), "invalid value found in confidences"

    assert torch.isfinite(avails).all(), "invalid value found in avails"



    # convert to (batch_size, num_modes, future_len, num_coords)

    gt = torch.unsqueeze(gt, 1)  # add modes

    avails = avails[:, None, :, None]  # add modes and cords



    # error (batch_size, num_modes, future_len)

    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability



    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it

        # error (batch_size, num_modes)

        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time



    # use max aggregator on modes for numerical stability

    # error (batch_size, num_modes)

    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one

    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes

    # print("error", error)

    return torch.mean(error)





def pytorch_neg_multi_log_likelihood_single(

    gt: Tensor, pred: Tensor, avails: Tensor

) -> Tensor:

    """



    Args:

        gt (Tensor): array of shape (bs)x(time)x(2D coords)

        pred (Tensor): array of shape (bs)x(time)x(2D coords)

        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep

    Returns:

        Tensor: negative log-likelihood for this example, a single float number

    """

    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)

    # create confidence (bs)x(mode=1)

    batch_size, future_len, num_coords = pred.shape

    confidences = pred.new_ones((batch_size, 1))

    return pytorch_neg_multi_log_likelihood_batch(gt, pred.unsqueeze(1), confidences, avails)


class LyftModel(pl.LightningModule):

    """Model is resnet101_02 pretrained on imagenet.

    We must replace the input and the final layer to address Lyft requirements.

    """



    def __init__(self, cfg: Dict, pretrained=True):

        super().__init__()



        self.cfg = cfg

        self.dm = LocalDataManager(None)

        self.rast = build_rasterizer(self.cfg, self.dm)



        

        self.backbone = resnet18(pretrained=True, progress=True)

        

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2

        num_in_channels = 3 + num_history_channels



        self.backbone.conv1 = nn.Conv2d(

            num_in_channels,

            self.backbone.conv1.out_channels,

            kernel_size=self.backbone.conv1.kernel_size,

            stride=self.backbone.conv1.stride,

            padding=self.backbone.conv1.padding,

            bias=False,

        )

        

        # This is 512 for resnet18 and resnet34;

        # And it is 2048 for the other resnets

        backbone_out_features = 512



        # X, Y coords for the future positions (output shape: Bx50x2)

        self.future_len = cfg["model_params"]["future_num_frames"]

        num_targets = 2 * self.future_len



        # You can add more layers here.

        self.head = nn.Sequential(

            # nn.Dropout(0.2),

            nn.Linear(in_features=backbone_out_features+36, out_features=4096),

        )

        self.num_preds = num_targets * 3

        self.num_modes = 3

        

        self.logit = nn.Linear(4096, out_features=self.num_preds + 3)





    def forward(self, x,y):

        x = self.backbone.conv1(x)

        x = self.backbone.bn1(x)

        x = self.backbone.relu(x)

        x = self.backbone.maxpool(x)



        x = self.backbone.layer1(x)

        x = self.backbone.layer2(x)

        x = self.backbone.layer3(x)

        x = self.backbone.layer4(x)



        x = self.backbone.avgpool(x)

        x = torch.flatten(x, 1)

        #y = self.meta(y)

        x = torch.cat((x, y), dim=1)

        

        x = self.head(x)

        x = self.logit(x)

        

        bs, _ = x.shape

        pred, confidences = torch.split(x, self.num_preds, dim=1)

        pred = pred.view(bs, self.num_modes, self.future_len, 2)

        assert confidences.shape == (bs, self.num_modes)

        confidences = torch.softmax(confidences, dim=1)

        

        return pred, confidences





    def training_step(self, batch, batch_idx):

        target_availabilities = torch.tensor(

            batch["target_availabilities"], device=self.device

        )

        targets = torch.tensor(batch["target_positions"], device=self.device)

        data = torch.tensor(batch["image"], device=self.device)

        meta=torch.cat((batch['extent']

                    ,torch.flatten(batch['history_yaws'].float(), 1)

                    ,batch['history_positions'][:,:,0].float(),batch['history_positions'][:,:,1].float()), dim=1).to(self.device)





        outputs,confidence = self(data,meta)

        loss = pytorch_neg_multi_log_likelihood_batch(targets, outputs, confidence, target_availabilities)



        pbar ={'train_loss':loss}

        return {'loss':loss,'progress_bar':pbar}



    def validation_step(self, batch, batch_idx):

        target_availabilities = torch.tensor(

            batch["target_availabilities"], device=self.device

        )

        targets = torch.tensor(batch["target_positions"], device=self.device)

        data = torch.tensor(batch["image"], device=self.device)

        meta=torch.cat((batch['extent']

                    ,torch.flatten(batch['history_yaws'].float(), 1)

                    ,batch['history_positions'][:,:,0].float(),batch['history_positions'][:,:,1].float()), dim=1).to(self.device)





        outputs,confidence = self(data,meta)

        loss = pytorch_neg_multi_log_likelihood_batch(targets, outputs, confidence, target_availabilities)





        return {'val_loss':loss}



    def configure_optimizers(self):

        return optim.Adam(self.parameters(), lr=1e-3)





#===== INIT DATASET

train_cfg = cfg["train_data_loader"]



# Rasterizer

rasterizer = build_rasterizer(cfg, dm)



# Train dataset/dataloader

train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()

train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

train_dataset = DataLoader(train_dataset,

                              shuffle=train_cfg["shuffle"],#shuffle=True

                              batch_size=train_cfg["batch_size"],#batch_size=24

                              num_workers=train_cfg["num_workers"])#num_workers=4







#print(train_dataset)



val_cfg = cfg["valid_data_loader"]



# Rasterizer

rasterizer = build_rasterizer(cfg, dm)



# Test dataset/dataloader

val_zarr = ChunkedDataset(dm.require(val_cfg["key"])).open()



val_dataset = AgentDataset(cfg, val_zarr, rasterizer)

val_dataset = DataLoader(val_dataset,

                             shuffle=val_cfg["shuffle"],

                             batch_size=val_cfg["batch_size"],

                             num_workers=val_cfg["num_workers"])





#print(val_dataloader)



model = LyftModel(cfg,pretrained=True)

trainer = pl.Trainer(tpu_cores=1, max_steps=500)#,



trainer.fit(model,train_dataset,val_dataset) #problem
