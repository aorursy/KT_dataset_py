!jupyter nbconvert --version

!papermill --version
# ensure version of L5Kit

import l5kit

assert l5kit.__version__ == "1.1.0"
import numpy as np

import os

import torch



from torch import nn, optim

from torch.utils.data import DataLoader

from torchvision.models.resnet import resnet50

from tqdm import tqdm

from typing import Dict



from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.geometry import transform_points

from l5kit.dataset import AgentDataset

from l5kit.evaluation import write_pred_csv

from l5kit.rasterization import build_rasterizer
def build_model(cfg: Dict) -> torch.nn.Module:

    # load pre-trained Conv2D model

    model = resnet50(pretrained=False)



    # change input channels number to match the rasterizer's output

    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2

    num_in_channels = 3 + num_history_channels

    model.conv1 = nn.Conv2d(

        num_in_channels,

        model.conv1.out_channels,

        kernel_size=model.conv1.kernel_size,

        stride=model.conv1.stride,

        padding=model.conv1.padding,

        bias=False,

    )

    # change output size to (X, Y) * number of future states

    num_targets = 2 * cfg["model_params"]["future_num_frames"]

    model.fc = nn.Linear(in_features=2048, out_features=num_targets)



    return model



def forward(data, model, device):

    inputs = data["image"].to(device)

    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)

    targets = data["target_positions"].to(device)

    # Forward pass

    outputs = model(inputs).reshape(targets.shape)

    return outputs
os.environ["L5KIT_DATA_FOLDER"] = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"

dm = LocalDataManager(None)



cfg = {

    'format_version': 4,

    'model_params': {

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

        'semantic_map_key': 'semantic_map/semantic_map.pb',

        'dataset_meta_key': 'meta.json',

        'filter_agents_threshold': 0.5

    },

    

    'test_data_loader': {

        'key': 'scenes/test.zarr',

        'batch_size': 12,

        'shuffle': False,

        'num_workers': 0

    }



}
# ===== INIT DATASET

test_cfg = cfg["test_data_loader"]



test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()

test_mask = np.load("/kaggle/input/lyft-motion-prediction-autonomous-vehicles/scenes/mask.npz")["arr_0"]



rasterizer = build_rasterizer(cfg, dm)

test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)

test_dataloader = DataLoader(test_dataset,

                             shuffle=test_cfg["shuffle"],

                             batch_size=test_cfg["batch_size"],

                             num_workers=test_cfg["num_workers"])

print(test_dataset)
# ==== INIT MODEL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = build_model(cfg).to(device)



model.load_state_dict(torch.load("/kaggle/input/baseline-weights/baseline_weights.pth", map_location=device))
# ==== EVAL LOOP

model.eval()

torch.set_grad_enabled(False)



# store information for evaluation

future_coords_offsets_pd = []

timestamps = []

agent_ids = []



progress_bar = tqdm(test_dataloader)

for data in progress_bar:

    

    # convert agent coordinates into world offsets

    agents_coords = forward(data, model, device).cpu().numpy().copy()

    world_from_agents = data["world_from_agent"].numpy()

    centroids = data["centroid"].numpy()

    coords_offset = []

    

    for agent_coords, world_from_agent, centroid in zip(agents_coords, world_from_agents, centroids):

        coords_offset.append(transform_points(agent_coords, world_from_agent) - centroid[:2])

    

    future_coords_offsets_pd.append(np.stack(coords_offset))

    timestamps.append(data["timestamp"].numpy().copy())

    agent_ids.append(data["track_id"].numpy().copy())
write_pred_csv("submission.csv",

               timestamps=np.concatenate(timestamps),

               track_ids=np.concatenate(agent_ids),

               coords=np.concatenate(future_coords_offsets_pd),

              )