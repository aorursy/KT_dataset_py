import gc

import os

from pathlib import Path

import random

import sys



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
import l5kit

from l5kit.evaluation.metrics import neg_multi_log_likelihood

print("l5kit version:", l5kit.__version__)
future_len = 50

n_coords = 2

n_modes = 3





gt = np.random.uniform(-1.0, 1.0, (future_len, n_coords))

pred = np.broadcast_to(gt[None, :, :], (n_modes, future_len, n_coords)) + np.random.uniform(-0.2, 0.2, (n_modes, future_len, n_coords))

confidences = np.random.uniform(0.0, 1.0, (n_modes,))

confidences /= np.sum(confidences)

avails = (np.random.uniform(0.0, 1.0, (future_len,)) > 0.3).astype(np.float64)
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py

import torch

from torch import Tensor





def pytorch_neg_multi_log_likelihood(

    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor

) -> Tensor:

    """

    Compute a negative log-likelihood for the multi-modal scenario.

    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:

    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    https://leimao.github.io/blog/LogSumExp/

    Args:

        gt (Tensor): array of shape (time)x(2D coords)

        pred (Tensor): array of shape (modes)x(time)x(2D coords)

        confidences (Tensor): array of shape (modes) with a confidence for each mode in each sample

        avails (Tensor): array of shape (time) with the availability for each gt timestep

    Returns:

        Tensor: negative log-likelihood for this example, a single float number

    """

    assert len(pred.shape) == 3, f"expected 3D (MxTxC) array for pred, got {pred.shape}"

    num_modes, future_len, num_coords = pred.shape



    assert gt.shape == (future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"

    assert confidences.shape == (num_modes,), f"expected 1D (Modes) array for gt, got {confidences.shape}"

    assert abs(torch.sum(confidences).item() - 1.0) < 1e-6, "confidences should sum to 1"

    assert avails.shape == (future_len,), f"expected 1D (Time) array for gt, got {avails.shape}"

    # assert all data are valid

    assert torch.isfinite(pred).all(), "invalid value found in pred"

    assert torch.isfinite(gt).all(), "invalid value found in gt"

    assert torch.isfinite(confidences).all(), "invalid value found in confidences"

    assert torch.isfinite(avails).all(), "invalid value found in avails"



    gt = torch.unsqueeze(gt, 0)  # add modes

    avails = avails[None, :, None]  # add modes and cords



    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability



    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it

        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time



    # use max aggregator on modes for numerical stability

    max_value = error.max()  # error are negative at this point, so max() gives the minimum one

    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1)) - max_value  # reduce modes

    return error
value_numpy = neg_multi_log_likelihood(gt, pred, confidences, avails)



value_torch = pytorch_neg_multi_log_likelihood(

    torch.tensor(gt),

    torch.tensor(pred),

    torch.tensor(confidences),

    torch.tensor(avails)

)



print("value_numpy: ", value_numpy)

print("value_torch: ", value_torch)