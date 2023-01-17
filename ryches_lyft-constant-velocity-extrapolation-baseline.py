# import os



# ## =====================================================================================

# ## This is a temporarly fix for the freezing and the cuda issues. You can add this

# ## utility script instead of kaggle_l5kit until Kaggle resolve these issues.

# ## 

# ## You will be able to train and submit your results, but not all the functionality of

# ## l5kit will work properly.



# ## More details here:

# ## https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/177125



# ## this script transports l5kit and dependencies

# os.system('pip install --target=/kaggle/working pymap3d==2.1.0')

# os.system('pip install --target=/kaggle/working protobuf==3.12.2')

# os.system('pip install --target=/kaggle/working transforms3d')

# os.system('pip install --target=/kaggle/working zarr')

# os.system('pip install --target=/kaggle/working ptable')



# os.system('pip install --no-dependencies --target=/kaggle/working l5kit')
from typing import Dict



import matplotlib.pyplot as plt

import numpy as np

import torch

# torch.multiprocessing.set_sharing_strategy('file_system')

from torch import nn, optim

from torch.utils.data import DataLoader

from tqdm.notebook import tqdm



from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.rasterization import build_rasterizer

from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, write_gt_csv

from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace

from l5kit.geometry import transform_points

from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory

from l5kit.dataset import AgentDataset

import os
DIR_INPUT = "../input/lyft-motion-prediction-autonomous-vehicles"
cfg = {

    'format_version': 4,

    'model_params': {

        'history_num_frames': 2,

        'history_step_size': 1,

        'history_delta_time': 0.1,

        'future_num_frames': 50,

        'future_step_size': 1,

        'future_delta_time': 0.1

    },

    

    'raster_params': {

        'raster_size': [300, 300],

        'pixel_size': [0.5, 0.5],

        'ego_center': [0.25, 0.5],

        'map_type': 'py_semantic',

        'satellite_map_key': 'aerial_map/aerial_map.png',

        'semantic_map_key': 'semantic_map/semantic_map.pb',

        'dataset_meta_key': 'meta.json',

        'filter_agents_threshold': 0.5

    },

    

    'test_data_loader': {

        'key': 'scenes/test.zarr',

        'batch_size': 8,

        'shuffle': False,

        'num_workers': 2

    },

    

    'train_data_loader': {

        'key': 'scenes/train.zarr',

        'batch_size': 8,

        'shuffle': True,

        'num_workers': 2

    },

    

    'val_data_loader': {

        'key': 'scenes/validate.zarr',

        'batch_size': 8,

        'shuffle': False,

        'num_workers': 2

    }



}
# set env variable for data

os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT

dm = LocalDataManager(None)
rasterizer = build_rasterizer(cfg, dm)



# Test dataset/dataloader

test_zarr = ChunkedDataset(dm.require(cfg['test_data_loader']["key"])).open()

test_mask = np.load(f"{DIR_INPUT}/scenes/mask.npz")["arr_0"]

test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)

test_dataloader = DataLoader(test_dataset,

                             shuffle=False,

                             batch_size=8,

                             num_workers=4)
val_zarr = ChunkedDataset(dm.require(cfg['val_data_loader']["key"])).open()

val_dataset = AgentDataset(cfg, val_zarr, rasterizer)

val_dataloader = DataLoader(val_dataset,

                             shuffle=False,

                             batch_size=8,

                             num_workers=4)
def make_preds(dataloader, pred_path, truth_path, batches = 100):

    timestamps = []

    agent_ids = []

    future_coords_offsets_pd = []

    truths = []

    target_availabilities = []

    for batch_idx, data in enumerate(tqdm(dataloader)):

        if batch_idx == batches:

            break

        timestamps.append(data["timestamp"])

        agent_ids.append(data["track_id"])



        u_point = data["history_positions"][:, :1, :].detach().cpu().numpy()

        pu_point = data["history_positions"][:, 1, :].detach().cpu().numpy()

        speed = (u_point[:, 0, :] - pu_point)

        predictions = np.ones((data["target_positions"].shape))*u_point

        

        for i in range(50):

            predictions[:, i, :] = predictions[:, i-1, :] + speed

        future_coords_offsets_pd.append(predictions)

        truths.append(data["target_positions"])

        target_availabilities.append(data["target_availabilities"])

    print(np.concatenate(future_coords_offsets_pd).shape, np.concatenate(truths).shape)

    write_pred_csv(pred_path,

           timestamps=np.concatenate(timestamps),

           track_ids=np.concatenate(agent_ids),

           coords=np.concatenate(future_coords_offsets_pd),

          )

    write_gt_csv(truth_path,

                 np.concatenate(timestamps),

                 np.concatenate(agent_ids),

                 np.concatenate(truths),

                 np.concatenate(target_availabilities)

          )
val_labels_path = "val_labels.csv"

val_predictions_path = "val_predictions.csv"
make_preds(val_dataloader, val_predictions_path, val_labels_path)
metrics = compute_metrics_csv(val_labels_path, val_predictions_path, [neg_multi_log_likelihood, time_displace])

for metric_name, metric_mean in metrics.items():

    print(metric_name, metric_mean)
make_preds(test_dataloader, "submission.csv", "test_dummy_labels.csv", batches = -1)