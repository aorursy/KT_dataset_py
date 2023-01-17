!pip install git+https://github.com/rwightman/pytorch-image-models.git

!pip install --target=/kaggle/working pymap3d==2.1.0 -q

!pip install --target=/kaggle/working strictyaml -q

!pip install --target=/kaggle/working protobuf==3.12.2 -q

!pip install --target=/kaggle/working transforms3d -q

!pip install --target=/kaggle/working zarr -q

!pip install --target=/kaggle/working ptable -q

!pip install --no-dependencies --target=/kaggle/working l5kit==1.1.0 --upgrade -q

!pip install pytorch-pfn-extras==0.3.1
from timm.models import vision_transformer

import torch

import l5kit, os

import torch.nn as nn

import numpy

import warnings;warnings.filterwarnings("ignore")

from l5kit.rasterization import build_rasterizer

from l5kit.configs import load_config_data

from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR

from tqdm import tqdm

from l5kit.geometry import transform_points

from collections import Counter

from l5kit.data import PERCEPTION_LABELS

from prettytable import PrettyTable

from ignite.engine import Events, Engine, create_supervised_trainer

import torch.optim as optim

import pytorch_pfn_extras.training.extensions as E

from pytorch_pfn_extras.training import IgniteExtensionsManager

from l5kit.evaluation.metrics import neg_multi_log_likelihood

cfg = load_config_data("../input/lyft-config-files/agent_motion_config.yaml")

os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"

model = vision_transformer.vit_small_resnet50d_s3_224(pretrained=True)
from l5kit.data import ChunkedDataset, LocalDataManager

from l5kit.dataset import EgoDataset, AgentDataset

dm = LocalDataManager()

train_cfg = cfg["train_data_loader"]

rasterizer = build_rasterizer(cfg, dm)

train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()

train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

train_dataloader = torch.utils.data.DataLoader(train_dataset,

                              shuffle=train_cfg["shuffle"],

                              batch_size=train_cfg["batch_size"],

                              num_workers=train_cfg["num_workers"])



val_cfg = cfg["val_data_loader"]

val_zarr = ChunkedDataset(dm.require(val_cfg["key"])).open()

val_dataset = AgentDataset(cfg, val_zarr, rasterizer)

val_dataset = torch.utils.data.Subset(val_dataset, range(0, 4000))

val_dataloader = torch.utils.data.DataLoader(val_dataset,

                              shuffle=train_cfg["shuffle"],

                              batch_size=train_cfg["batch_size"],

                              num_workers=train_cfg["num_workers"])
class LyftVIT(nn.Module):

    

    def __init__(self, vit: nn.Module):

        super(LyftVIT, self).__init__()

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2

        num_in_channels = 3 + num_history_channels

        self.vit = vit

        num_targets = 2 * cfg["model_params"]["future_num_frames"]

        self.future_len = cfg["model_params"]["future_num_frames"]

        self.vit.patch_embed.backbone.conv1[0] = nn.Conv2d(

            num_in_channels,

            32,

            kernel_size=self.vit.patch_embed.backbone.conv1[0].kernel_size,

            stride=self.vit.patch_embed.backbone.conv1[0].stride,

            padding=self.vit.patch_embed.backbone.conv1[0].padding,

            bias=False,

        )

        

        

        self.num_preds = num_targets * 3

        self.num_modes = 3

        

        self.logit = nn.Linear(1000, out_features=self.num_preds + self.num_modes)

        

    def forward(self, x):

        x = self.vit(x)

        x = torch.flatten(x, 1)

        x = self.logit(x)

        bs, _ = x.shape

        pred, confidences = torch.split(x, self.num_preds, dim=1)

        pred = pred.view(bs, self.num_modes, self.future_len, 2)

        assert confidences.shape == (bs, self.num_modes)

        confidences = torch.softmax(confidences, dim=1)

        return pred, confidences
model = LyftVIT(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.MSELoss(reduction="none")
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
def train_step(engine, batch):

    model.train()

    optimizer.zero_grad()

    x, y = batch["image"].to(device), batch["target_positions"].to(device)

    avails = batch["target_availabilities"].unsqueeze(-1).to(device)

    y_pred, conf = model(x)

    loss = pytorch_neg_multi_log_likelihood_batch(y, y_pred, conf, avails[:, :, 0])

    loss.backward()

    optimizer.step()

    return loss.item()



trainer = Engine(train_step)



def validation_step(image, target_positions, **kwargs):

    x, y = image.to(device), target_positions.to(device)

    y_pred = model(x)

    return y_pred, y
valid_evaluator = E.Evaluator(

    val_dataloader,

    model,

    progress_bar=False,

    eval_func=validation_step,

)



log_trigger = (1000, "iteration")

log_report = E.LogReport(trigger=log_trigger)

extensions = [

    log_report,  # Save `log` to file

    valid_evaluator,

    # E.FailOnNonNumber()  # Stop training when nan is detected.

    E.ProgressBarNotebook(update_interval=100),  # Show progress bar during training

    E.PrintReportNotebook(),  # Show "log" on jupyter notebook  

]
models = {"main": model}

optimizers = {"main": optimizer}

manager = IgniteExtensionsManager(

    trainer,

    models,

    optimizers,

    7,

    extensions=extensions,

    out_dir="../working",

)

manager.extend(E.snapshot_object(model, "predictor.pt"),

               trigger=(50, "iteration")) 
manager.iteration = 0

manager._iters_per_epoch = len(train_dataloader)

trainer.run(train_dataloader, max_epochs=7)