# torch imports
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101
import torch.functional as F

# l5kit imports
import l5kit
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory

# common imports
import os
import random
import time
import pandas as pd
from typing import Dict
from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
l5kit.__version__
torch.cuda.is_available()
# --- Function utils ---
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py
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
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
set_seed(42)
def resnet_forward(backbone, x):    
    #with torch.set_grad_enabled(False):
    with torch.no_grad():
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)

        x = backbone.avgpool(x)
        x = torch.flatten(x, 1)
    return x
def find_no_of_trainable_params(model):
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(total_trainable_params)
    return total_trainable_params
def LSTM_batch_transform(image_data, base_model):    
    
    BATCH_SIZE = image_data.shape[0]
    
    """ LANES, TRAFFIC LIGHT DATA ENCODING """
    infra_data = image_data[:, -3:, :, :]
    infra_data = resnet_forward(base_model, infra_data)
    infra_data = torch.repeat_interleave(infra_data.unsqueeze(1), NUMBER_OF_HISTORY_FRAMES, dim=1)
    #print(infra_data.shape)
    
    """ EGO, AGENT VEHICLE DATA ENCODING """
    # agent frames
    agent_data = image_data[:, 0:NUMBER_OF_HISTORY_FRAMES, :, :]
    #print(agent_data.shape)

    # ego vehicle frames
    ego_data = image_data[:, NUMBER_OF_HISTORY_FRAMES:-3, :,:]
    #print(ego_data.shape)

    # combined ego and agent frames, duplicating across 3 channels
    vehicle_data = torch.repeat_interleave(ego_data + agent_data, 3, dim=1)

    # pretrained model requires (batch_size, 3, 224, 224), hence reshaping
    vehicle_data = vehicle_data.view(-1, 3, RASTER_IMG_SIZE, RASTER_IMG_SIZE)
    #print(vehicle_data.shape)

    # passing through model and reshaping
    history_vehicle_data = resnet_forward(base_model, vehicle_data)
    history_vehicle_data =  history_vehicle_data.view(BATCH_SIZE, -1, 512)
    #print(history_vehicle_data.shape)
    
    """concatenating history_vehicle_data and infra_data """
    LSTM_input = torch.cat((history_vehicle_data, infra_data), dim=-1)
    #print(f'LSTM input shape is {temp.shape}')
    
    return LSTM_input
def forward(data, model, hidden_state, device, criterion = pytorch_neg_multi_log_likelihood_batch):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device)
    batch_size = inputs.shape[0]
    
    # converting image data to sequential data for LSTM model
    LSTM_input = LSTM_batch_transform(inputs, encoding_model)
    
    # LSTM model prediction and confidence
    prediction, hidden_state = model(LSTM_input, hidden_state)
    hidden_state = (hidden_state[0].data, hidden_state[1].data)
    prediction, confidences = torch.split(prediction, 300, dim=1)
    prediction = prediction.view(batch_size, 3, 50, 2)
    confidences = torch.softmax(confidences, dim=1)
    
    # calculating NLL loss 
    loss = pytorch_neg_multi_log_likelihood_batch(targets, prediction, confidences, target_availabilities)
    
    return loss, hidden_state, prediction, confidences
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, time_steps, use_LSTM = False):
        super(RNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps
        
        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        
        if use_LSTM == True:
            self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        
        
        # last, fully-connected layer
        self.fc = nn.Linear(time_steps * hidden_dim, output_size)

    def forward(self, x, hidden):
        # x (batch_size, time_step, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)
        
        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)
        # shape output to be (batch_size*time_step, hidden_dim)
        r_out = r_out.reshape(batch_size,-1)  
        
        # get final output 
        output = self.fc(r_out)
        
        return output, hidden
# --- Lyft configs ---
cfg = {
    'format_version': 4,
    'data_path': "/kaggle/input/lyft-motion-prediction-autonomous-vehicles/",
    'model_params': {
        'model_architecture': 'LSTM',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'model_name': "LSTM_baseline_r34",
        'weight_path': "/kaggle/input/lstm-baseline-weights/LSTM_baseline_r34_9750.pth",
        'lr': 1e-3,
        'train': True,
        'predict': False
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
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 4
    },
    
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 4
    },

    'sample_data_loader': {
        'key': 'scenes/sample.zarr',
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 4
    },

    'train_params': {
        'train_start_index' : 9751,
        'max_num_steps': 12002,
        'checkpoint_every_n_steps': 500,
    }
}
NUMBER_OF_HISTORY_FRAMES = cfg['model_params']['history_num_frames'] + 1
RASTER_IMG_SIZE = cfg['raster_params']['raster_size'][0]
NUM_MODES = 3
NUMBER_OF_FUTURE_FRAMES = cfg['model_params']['future_num_frames']

### TRAIN FROM WHERE LEFT OFF, CHANGE THE STARTING INDICES VARIABLE ACCORDINGLY
TRAIN_START_INDICES = cfg['train_params']['train_start_index']
# set env variable for data
DIR_INPUT = cfg["data_path"]
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)
rasterizer = build_rasterizer(cfg, dm)
# ===== INIT TRAIN DATASET============================================================
train_cfg = cfg["train_data_loader"]
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
print('Length of Train dataset is ' ,len(train_dataset))
print("==================================TRAIN DATA==================================")
print(train_dataset)
len(train_dataset)
sampled_indices = np.random.choice(len(train_dataset), size = len(train_dataset), replace = False)
print('Before slicing, start indices are ', sampled_indices[0:10])
TRAIN_START_INDICES
sampled_indices = sampled_indices[TRAIN_START_INDICES:]
print('After slicing, start indices are ', sampled_indices[0:10])
Datasampler = SubsetRandomSampler(sampled_indices)
train_dataloader = DataLoader(train_dataset, sampler=Datasampler, batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"])
# ==== INIT MODEL=================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device {device}')
encoding_model = resnet34(pretrained=True)
encoding_model.to(device);

# Freeze parameters so we don't backprop through them
for param in encoding_model.parameters():
    param.requires_grad = False

Total_trainable_params = find_no_of_trainable_params(encoding_model)
print(f'There are {Total_trainable_params} trainable parameters in the model')

# set to evaluation mode
encoding_model.eval();
# decide on hyperparameters
input_size   = 1024 
output_size  = 303
hidden_dim   = 64
n_layers     = 2
# instantiate an RNN
model = RNN(input_size, output_size, hidden_dim, n_layers, 11, use_LSTM=True)
model.to(device)
#print(LSTM_baseline_model)

total_params = find_no_of_trainable_params(model)
print(f'There are {total_params} parameters in the LSTM model')
## loading the pretrained weights
model.load_state_dict(torch.load(cfg['model_params']['weight_path']))
## Adam optimiser function
optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])
# ==== TRAINING LOOP =========================================================
if cfg["model_params"]["train"]:
    
    tr_it = iter(train_dataloader)
    progress_bar = tqdm(range(TRAIN_START_INDICES, 
                              TRAIN_START_INDICES + cfg["train_params"]["max_num_steps"]))
    num_iter = cfg["train_params"]["max_num_steps"]
    losses_train = []
    iterations = []
    metrics = []
    times = []
    model_name = cfg["model_params"]["model_name"]
    start = time.time()
    hidden_state = None
    
    for i in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            data = next(tr_it)
            
        # Forward pass
        model.train()
        torch.set_grad_enabled(True)
        loss, hidden_state, _, _ = forward(data, model, hidden_state, device)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())

        progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")
        if i % cfg['train_params']['checkpoint_every_n_steps'] == 0:
            torch.save(model.state_dict(), f'{model_name}_{i + TRAIN_START_INDICES}.pth')
            iterations.append(i)
            metrics.append(np.mean(losses_train))
            times.append((time.time()-start)/60)
    
    results = pd.DataFrame({'iterations': iterations, 'metrics (avg)': metrics, 'elapsed_time (mins)': times})
    results.to_csv(f"train_metrics_{model_name}_{num_iter}.csv", index = False)
    train_losses_csv = pd.DataFrame({'iteration': TRAIN_START_INDICES + np.arange(len(losses_train)), 
                                 'losses_train': losses_train})
    train_losses_csv.to_csv(f"train_losses_{model_name}_{num_iter}.csv", index = False)
    print(f"Total training time is {(time.time()-start)/60} mins")
    print(results.head())