!pip install --no-index -q --use-feature=2020-resolver -f ../input/kaggle-l5kit-110 l5kit
import gc

import os

from pathlib import Path

import random

import sys

from l5kit.data import ChunkedDataset, LocalDataManager

from l5kit.dataset import EgoDataset, AgentDataset

from tqdm.notebook import tqdm

import numpy as np

import pandas as pd

import scipy as sp

import matplotlib.pyplot as plt

import seaborn as sns



from IPython.core.display import display, HTML



# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

import plotly.io as pio

pio.templates.default = "plotly_dark"



# --- models ---

from sklearn import preprocessing

from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

import catboost as cb



# --- setup ---

pd.set_option('max_columns', 50)


from l5kit.data import ChunkedDataset, LocalDataManager

from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.evaluation import write_pred_csv

from l5kit.rasterization import build_rasterizer

from l5kit.configs import load_config_data

from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR

from l5kit.geometry import transform_points
import torch

from pathlib import Path

# !pip install pytorch_pfn_extras

# import pytorch_pfn_extras as ppe

from math import ceil

# from pytorch_pfn_extras.training import IgniteExtensionsManager

# from pytorch_pfn_extras.training.triggers import MinValueTrigger

from torch import nn, optim

from torch.utils.data import DataLoader

from torch.utils.data.dataset import Subset

# import pytorch_pfn_extras.training.extensions as E
# --- Dataset utils ---

from typing import Callable



from torch.utils.data.dataset import Dataset



class TransformDataset(Dataset):

    def __init__(self, dataset: Dataset, transform: Callable):

        self.dataset = dataset

        self.transform = transform



    def __getitem__(self, index):

        batch = self.dataset[index]

        return self.transform(batch)



    def __len__(self):

        return len(self.dataset)
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
# --- Model utils ---

import torch

from torchvision.models import resnet18, resnet50

from torch import nn

from typing import Dict

import torch.nn.functional as F



class LyftMultiModelAttn(nn.Module):



    def __init__(self, cfg: Dict, num_modes=3):

        super().__init__()

        

        backbone = resnet18(pretrained=True, progress=True)

        self.backbone = backbone



        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2

        num_in_channels = 3 + num_history_channels



        self.backbone.conv1 = nn.Conv2d(

            1,

            self.backbone.conv1.out_channels,

            kernel_size=self.backbone.conv1.kernel_size,

            stride=self.backbone.conv1.stride,

            padding=self.backbone.conv1.padding,

            bias=False,

        )

        # This is 512 for resnet18 and resnet34;

        # And it is 2048 for the other resnets

        backbone_out_features = 512        

        self.backbone.layer5 = nn.Conv2d(

            backbone_out_features,

            128,

            kernel_size=2,

            stride=2,

            # padding=self.backbone.conv1.padding,

            bias=False,

        )        



#         self.backbone.layer6 = nn.Conv2d(

#             1024,

#             2048,

#             kernel_size=2,

#             stride=2,

#             # padding=self.backbone.conv1.padding,

#             bias=False,

#         )        



#         self.backbone.layer7 = nn.Conv2d(

#             2048,

#             2048,

#             kernel_size=2,

#             stride=2,

#             # padding=self.backbone.conv1.padding,

#             bias=False,

#         )        

        





        # X, Y coords for the future positions (output shape: batch_sizex50x2)

        self.future_len = cfg["model_params"]["future_num_frames"]

        num_targets = 2 * self.future_len



        # You can add more layers here.

        self.head = nn.Sequential(

            # nn.Dropout(0.2),

            nn.Linear(in_features=128*7*7, out_features=4096),

        )



        self.num_preds = num_targets * num_modes

        self.num_modes = num_modes

        self.attn_layer = nn.Sequential(

            nn.Linear(128, 512, False),

            nn.BatchNorm1d(512),

            nn.Tanh(),          

            # nn.Dropout(0.5),

            nn.Linear(512, 1, False)

        )

        self.logit = nn.Linear(1024, out_features=self.num_preds + num_modes)



        num_layers = 1

        self.lstm = nn.LSTM(input_size=128*7*7,

                            hidden_size=1024,

                            num_layers=num_layers)



        self.bs = cfg['train_data_loader']['batch_size']



    def forward(self, x):



        batch_size, time_steps, height, width = x.size()

        x = x.view(batch_size * time_steps, 1, height, width)



        x = self.backbone.conv1(x)

        x = self.backbone.bn1(x)

        x = self.backbone.relu(x)

        # x = self.backbone.maxpool(x)



        x = self.backbone.layer1(x)

        x = self.backbone.layer2(x)

        x = self.backbone.layer3(x)

        x = self.backbone.layer4(x)

        x = self.backbone.layer5(x)

        # x = self.backbone.layer6(x)

        # x = self.backbone.layer7(x)

        _, _, height, width = x.size()

        x = x.view(batch_size * height * width * time_steps, 128)

        alpha = self.attn_layer(x)

        alpha = alpha.view(batch_size * time_steps, height * width)

        alpha = F.softmax(alpha, dim=1)

        alpha = alpha.view(batch_size * height * width * time_steps, 1).clone().repeat(1, 128)

        x = x * alpha

        # x = x.view(batch_size, time_steps, height * width *128)





        # x = self.backbone.avgpool(x)

        # x = torch.flatten(x, 1)



        # x = self.head(x)

        

        x = x.view(batch_size, time_steps, -1)



        x = x.permute(1, 0, 2)



        _, (x, _) = self.lstm(x)

        x = x.squeeze()

        del _

        torch.cuda.empty_cache()

        # x = torch.flatten(x, 1)

        x = self.logit(x)



        # pred (bs)x(modes)x(time)x(2D coords)

        # confidences (bs)x(modes)

        bs, _ = x.shape

        pred, confidences = torch.split(x, self.num_preds, dim=1)

        pred = pred.view(bs, self.num_modes, self.future_len, 2)

        assert confidences.shape == (bs, self.num_modes)

        confidences = torch.softmax(confidences, dim=1)

        return pred, confidences
class LyftMultiRegressor(nn.Module):

    """Single mode prediction"""



    def __init__(self, predictor, lossfun=pytorch_neg_multi_log_likelihood_batch):

        super().__init__()

        self.predictor = predictor

        self.lossfun = lossfun



    def forward(self, image, targets, target_availabilities):

        pred, confidences = self.predictor(image)

        loss = self.lossfun(targets, pred, confidences, target_availabilities)

        metrics = {

            "loss": loss.item(),

            "nll": pytorch_neg_multi_log_likelihood_batch(targets, pred, confidences, target_availabilities).item()

        }

        # ppe.reporting.report(metrics, self)

        return loss, metrics
def run_prediction(predictor, data_loader):

    predictor.eval()



    pred_coords_list = []

    confidences_list = []

    timestamps_list = []

    track_id_list = []



    with torch.no_grad():

        dataiter = tqdm(data_loader)

        for data in dataiter:

            image = data["image"].to(device)

            # target_availabilities = data["target_availabilities"].to(device)

            # targets = data["target_positions"].to(device)

            pred, confidences = predictor(image)



            pred_coords_list.append(pred.cpu().numpy().copy())

            confidences_list.append(confidences.cpu().numpy().copy())

            timestamps_list.append(data["timestamp"].numpy().copy())

            track_id_list.append(data["track_id"].numpy().copy())

    timestamps = np.concatenate(timestamps_list)

    track_ids = np.concatenate(track_id_list)

    coords = np.concatenate(pred_coords_list)

    confs = np.concatenate(confidences_list)

    return timestamps, track_ids, coords, confs
# --- Training utils ---

from ignite.engine import Engine

def create_trainer(model, optimizer, device) -> Engine:

    model.to(device)

    def update_fn(engine, batch):

        model.train()

        optimizer.zero_grad()

        loss, metrics = model(*[elem.to(device) for elem in batch])

        loss.backward()

        optimizer.step()

        return metrics

    trainer = Engine(update_fn)

    return trainer
# --- Utils ---

import yaml





def save_yaml(filepath, content, width=120):

    with open(filepath, 'w') as f:

        yaml.dump(content, f, width=width)





def load_yaml(filepath):

    with open(filepath, 'r') as f:

        content = yaml.safe_load(f)

    return content



class DotDict(dict):

    """dot.notation access to dictionary attributes



    Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767

    """  # NOQA



    __getattr__ = dict.get

    __setattr__ = dict.__setitem__

    __delattr__ = dict.__delitem__
# --- Lyft configs ---

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

        'raster_size': [224, 224],

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

        'batch_size':5,

        'shuffle': True,

        'num_workers': 4 

    },



    'valid_data_loader': {

        'key': 'scenes/validate.zarr',

        'batch_size': 5,

        'shuffle': False,

        'num_workers': 4

    },

    'test_data_loader': {

        'key': 'scenes/test.zarr',

        'batch_size': 5,

        'shuffle': False,

        'num_workers': 4

    },

    'train_params': {

        'max_num_steps': 10000,

        'checkpoint_every_n_steps': 5000,



        # 'eval_every_n_steps': -1

    }

}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



flags_dict = {

    "debug": True,

    # --- Data configs ---

    "l5kit_data_folder": "/kaggle/input/lyft-motion-prediction-autonomous-vehicles",

    # --- Model configs ---

    "pred_mode": "multi",

    # --- Training configs ---

    "device": device,

    "out_dir": "results/multi_train",

    "epoch": 20,

    "snapshot_freq": 50,

}

print(device)
flags = DotDict(flags_dict)

out_dir = Path(flags.out_dir)

os.makedirs(str(out_dir), exist_ok=True)

print(f"flags: {flags_dict}")

save_yaml(out_dir / 'flags.yaml', flags_dict)

save_yaml(out_dir / 'cfg.yaml', cfg)

debug = flags.debug
# set env variable for data

os.environ["L5KIT_DATA_FOLDER"] = flags.l5kit_data_folder

dm = LocalDataManager(None)



print("Load dataset...")

train_cfg = cfg["train_data_loader"]

valid_cfg = cfg["valid_data_loader"]



# Rasterizer

rasterizer = build_rasterizer(cfg, dm)



# Train dataset/dataloader

def transform(batch):

    return batch["image"], batch["target_positions"], batch["target_availabilities"]



train_path = "scenes/sample.zarr" if debug else train_cfg["key"]

train_zarr = ChunkedDataset(dm.require(train_path)).open()

print("train_zarr", type(train_zarr))

train_agent_dataset = AgentDataset(cfg, train_zarr, rasterizer)

train_dataset = TransformDataset(train_agent_dataset, transform)

if debug:

    # Only use 1000 dataset for fast check...

    train_dataset = Subset(train_dataset, np.arange(1400))

else:

    train_dataset = Subset(train_dataset, np.arange(100000))



print(train_agent_dataset)



valid_path = "scenes/sample.zarr" if debug else valid_cfg["key"]

valid_zarr = ChunkedDataset(dm.require(valid_path)).open()

print("valid_zarr", type(train_zarr))

valid_agent_dataset = AgentDataset(cfg, valid_zarr, rasterizer)

valid_dataset = TransformDataset(valid_agent_dataset, transform)

if debug:

    # Only use 100 dataset for fast check...

    valid_dataset = Subset(valid_dataset, np.arange(100))

else:

    # Only use 1000 dataset for fast check...

    valid_dataset = Subset(valid_dataset, np.arange(1000))



print(valid_agent_dataset)

print("# AgentDataset train:", len(train_agent_dataset), "#valid", len(valid_agent_dataset))



# AgentDataset train: 22496709 #valid 21624612

# ActualDataset train: 100 #valid 100
train_cfg = cfg["train_data_loader"]

valid_cfg = cfg["valid_data_loader"]

train_loader = DataLoader(train_dataset,

                          shuffle=train_cfg["shuffle"],

                          batch_size=train_cfg["batch_size"],

                          num_workers=train_cfg["num_workers"])

valid_loader = DataLoader(

    valid_dataset,

    shuffle=valid_cfg["shuffle"],

    batch_size=valid_cfg["batch_size"],

    num_workers=valid_cfg["num_workers"]

)

print("# ActualDataset train:", len(train_dataset), "#valid", len(valid_dataset))

# set env variable for data

l5kit_data_folder = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"

os.environ["L5KIT_DATA_FOLDER"] = l5kit_data_folder

dm = LocalDataManager(None)



print("Load dataset...")

default_test_cfg = {

    'key': 'scenes/test.zarr',

    'batch_size': 32,

    'shuffle': False,

    'num_workers': 4

}

test_cfg = cfg.get("test_data_loader", default_test_cfg)



# Rasterizer

rasterizer = build_rasterizer(cfg, dm)



test_path = test_cfg["key"]

print(f"Loading from {test_path}")

test_zarr = ChunkedDataset(dm.require(test_path)).open()

print("test_zarr", type(test_zarr))

test_mask = np.load(f"{l5kit_data_folder}/scenes/mask.npz")["arr_0"]

test_agent_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)

test_dataset = test_agent_dataset

# if debug:

    # Only use 100 dataset for fast check...

    # test_dataset = Subset(test_dataset, np.arange(100))

test_loader = DataLoader(

    test_dataset,

    shuffle=test_cfg["shuffle"],

    batch_size=test_cfg["batch_size"],

    num_workers=test_cfg["num_workers"],

    pin_memory=True,

)



print(test_agent_dataset)

print("# AgentDataset test:", len(test_agent_dataset))

print("# ActualDataset test:", len(test_dataset))
device = torch.device(flags.device)



if flags.pred_mode == "multi":



    predictor = LyftMultiModelAttn(cfg)

  

    model = LyftMultiRegressor(predictor)



else:

    raise ValueError(f"[ERROR] Unexpected value flags.pred_mode={flags.pred_mode}")



model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.ExponentialLR(

    optimizer, gamma=0.99999)
# Train setup

trainer = create_trainer(model, optimizer, device)

MAX = 1e16 + 7

def eval_func(*batch):

    loss, metrics = model(*[elem.to(device) for elem in batch])

    return loss, metrics



def eval(model, loader, eval_func):

    

    model.eval()

    error = 0

    count = 0

    for batch_i, batch in enumerate(loader):

        count += 1

        with torch.no_grad():

            loss, metrics = model(*[elem.to(device) for elem in batch])

        error += loss.item()



        del metrics

        torch.cuda.empty_cache()

    print("Validation loss per batch {}".format(error/count))

    return loss



def train(model, loader, eval_func, optimizer):

    model.train()

    error = 0

    count = 0

    lastcheckpoint = flags.out_dir+'/intermediate_model.pth'

    if os.path.isfile(lastcheckpoint):

        print("loading ...")

        t = torch.load(lastcheckpoint, map_location=lambda storage, loc: storage)   

        model.predictor.load_state_dict(t['state_dict'])

        print("done")

    else:

        print("file not found")

        

    for batch in tqdm(loader):



        count += 1

        optimizer.zero_grad()        

        loss, metrics = model(*[elem.to(device) for elem in batch])

        loss.backward()

        optimizer.step()        

        scheduler.step()

        del metrics

        torch.cuda.empty_cache()        

        if count%10 == 0:

            print("saving at ", flags.out_dir+'/intermediate_model.pth')

            torch.save({'count': count, 'state_dict': model.predictor.state_dict()},

                       flags.out_dir+'/intermediate_model.pth')

            print("Epoch no. {} TR loss {} lr {}".format(count, error/count, optimizer.param_groups[0]['lr']))       

        error += loss.item()

        del loss

        torch.cuda.empty_cache()    

    print("training loss per batch {}".format(error/count))

    return loss





epoch = flags.epoch

for epoch_n in range(epoch):

    print("epoch no.", epoch_n)

    tl = train(model, train_loader, eval_func, optimizer)

    vl = eval(model, valid_loader, eval_func)

    if vl < MAX:

        timestamps, track_ids, coords, confs = run_prediction(model.predictor, test_loader)

        def saving_csv():

            csv_path = "submission.csv"

            write_pred_csv(

                csv_path,

                timestamps=timestamps,

                track_ids=track_ids,

                coords=coords,

                confs=confs)

            print(f"Saved to {csv_path}")

        saving_csv()

    basename = "epoch {} train loss {} val loss{}".format(epoch_n, tl, vl)

    torch.save({'epoch': epoch_n, 'state_dict': model.predictor.state_dict()},

               flags.out_dir+'/model.pth')