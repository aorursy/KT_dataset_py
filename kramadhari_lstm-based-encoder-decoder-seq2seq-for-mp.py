import numpy as np

import os
import torch
torch.manual_seed(0)

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18,resnet50,resnet101
from tqdm import tqdm
from typing import Dict
from torch import functional as F

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path
DIR_INPUT = "..input/lyft-motion-prediction-autonomous-vehicles/"
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)
VALIDATION = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = {
    'format_version': 4,
    'model_params': {
        'model_architecture': 'resnet18',
        
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },
    
    'raster_params': {
        'raster_size': [1, 1],
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
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 4
    },
    
    'val_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 4
    },
    
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 4
    },
    
    'train_params': {
        'checkpoint_every_n_steps': 5000,
        'max_num_steps': 10000,
        'eval_every_n_steps': 500

        
    }
}
# ===== INIT DATASET
train_cfg = cfg["train_data_loader"]

# Rasterizer
rasterizer = build_rasterizer(cfg, dm)

# Train dataset/dataloader
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
train_dataloader = DataLoader(train_dataset,
                              shuffle=train_cfg["shuffle"],
                              batch_size=train_cfg["batch_size"])
                              #num_workers=train_cfg["num_workers"])

print(train_dataset)
# ===== INIT  VAL DATASET
val_cfg = cfg["val_data_loader"]

# Rasterizer
rasterizer = build_rasterizer(cfg, dm)

# Train dataset/dataloader
val_zarr = ChunkedDataset(dm.require(val_cfg["key"])).open()
val_dataset = AgentDataset(cfg, val_zarr, rasterizer)
val_dataloader = DataLoader(val_dataset,
                              shuffle=val_cfg["shuffle"],
                              batch_size=val_cfg["batch_size"])
                              #num_workers=train_cfg["num_workers"])

print(val_dataset)
class EncoderLSTM_LyftModel(nn.Module):
    
    def __init__(self, cfg):
        super(EncoderLSTM_LyftModel, self).__init__()
        
        self.input_sz  = 2
        self.hidden_sz = 128
        self.num_layer = 1
        self.sequence_length = 11        
        
        self.Encoder_lstm = nn.LSTM(self.input_sz,self.hidden_sz,self.num_layer,batch_first=True)
       
    def forward(self,inputs):
        
        output,hidden_state = self.Encoder_lstm(inputs)
        
        return output,hidden_state
    
class DecoderLSTM_LyftModel(nn.Module):
    def __init__(self, cfg):
        super(DecoderLSTM_LyftModel, self).__init__()
        
        self.input_sz  = 128 #(2000 from fcn_en_output reshape to 50*40)
        self.hidden_sz = 128
        self.hidden_sz_en = 128
        self.num_layer = 1
        self.sequence_len_de = 1
        
        self.interlayer = 256

        num_targets = 2 * cfg["model_params"]["future_num_frames"]
        
        self.encoderLSTM = EncoderLSTM_LyftModel (cfg)

        
        self.Decoder_lstm = nn.LSTM( self.input_sz,self.hidden_sz,self.num_layer,batch_first=True)

        
        self.fcn_en_state_dec_state= nn.Sequential(nn.Linear(in_features=self.hidden_sz_en, out_features=self.interlayer),
                            nn.ReLU(inplace=True),
                            nn.Linear(in_features=self.interlayer, out_features=num_targets))

    def forward(self,inputs):
        
        _,hidden_state = self.encoderLSTM(inputs)
        
        inout_to_dec = torch.ones(inputs.shape[0],self.sequence_len_de,self.input_sz).to(device)

        #for i in range(cfg["model_params"]["future_num_frames"]+1): # this can be used to feed output from previous LSTM to anther one which is stacked.
        inout_to_dec,hidden_state   = self.Decoder_lstm(inout_to_dec,(hidden_state[0],hidden_state[1]) )          
        
        fc_out = self.fcn_en_state_dec_state (inout_to_dec.squeeze(dim=0))
        
        return fc_out.reshape(inputs.shape[0],cfg["model_params"]["future_num_frames"],-1)
# ==== INIT MODEL
model = DecoderLSTM_LyftModel(cfg)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=7000,gamma=0.1)
criterion = nn.MSELoss(reduction="none")
# ==== TRAIN LOOP
tr_it = iter(train_dataloader)
vl_it = iter(val_dataloader)

progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
losses_train = []
losses_mean_train = []
losses_val = []
losses_mean_val = []

for itr in progress_bar:
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_dataloader)
        data = next(tr_it)
    model.train()
    torch.set_grad_enabled(True)

    # Forward pass
    history_positions = data['history_positions'].to(device)
    history_availabilities = data['history_availabilities'].to(device)
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    targets_position = data["target_positions"].to(device)

    outputs = model(history_positions)

    loss = criterion(outputs,targets_position)
    # not all the output steps are valid, but we can filter them out from the loss using availabilities
    loss = loss * target_availabilities
    loss = loss.mean()

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    
    losses_train.append(loss.item())
    losses_mean_train.append(np.mean(losses_train))
    
    # Validation
    if VALIDATION :
        with torch.no_grad():
            try:
                val_data = next(vl_it)
            except StopIteration:
                vl_it = iter(val_dataloader)
                val_data = next(vl_it)

            model.eval()
            # Forward pass
            target_availabilities_val = val_data["target_availabilities"].unsqueeze(-1).to(device)
            targets_val = val_data["target_positions"].to(device)
            history_positions_val = val_data['history_positions'].to(device)
            history_availabilities_val = data['history_availabilities'].to(device)

            outputs_val = model(history_positions_val)
                    
            loss_v = criterion(outputs_val,targets_val)
            # not all the output steps are valid, but we can filter them out from the loss using availabilities
            loss_v = loss_v * target_availabilities_val
            loss_v = loss_v.mean()

            losses_val.append(loss_v.item())

            losses_mean_val.append(np.mean(losses_val))


        desc = f" TrainLoss: {round(loss.item(), 4)} ValLoss: {round(loss_v.item(), 4)} TrainMeanLoss: {np.mean(losses_train)} ValMeanLoss: {np.mean(losses_val)}" 
    else:
        desc = f" TrainLoss: {round(loss.item(), 4)}"


        #if len(losses_train)>0 and loss < min(losses_train):
        #    print(f"Loss improved from {min(losses_train)} to {loss}")
    lr_scheduler.step()

    progress_bar.set_description(desc)
# ===== INIT DATASET
test_cfg = cfg["test_data_loader"]

# Rasterizer
rasterizer = build_rasterizer(cfg, dm)

# Test dataset/dataloader
test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()
test_mask = np.load(f"{DIR_INPUT}/scenes/mask.npz")["arr_0"]
test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
test_dataloader = DataLoader(test_dataset,
                             shuffle=test_cfg["shuffle"],
                             batch_size=test_cfg["batch_size"],
                             num_workers=test_cfg["num_workers"])


print(test_dataloader)
model.eval()

future_coords_offsets_pd = []
timestamps = []
agent_ids = []

with torch.no_grad():
    dataiter = tqdm(test_dataloader)
    
    for data in dataiter:

        history_positions = data['history_positions'].to(device)

        outputs = model(history_positions)
        
        future_coords_offsets_pd.append(outputs.cpu().numpy().copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy())

write_pred_csv('submission.csv',
               timestamps=np.concatenate(timestamps),
               track_ids=np.concatenate(agent_ids),
               coords=np.concatenate(future_coords_offsets_pd))