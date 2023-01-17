!pip install --no-index -q --use-feature=2020-resolver -f ../input/kaggle-l5kit-110 l5kit 
!nvcc --version
import l5kit

import torch

import torchvision

l5kit.__version__, torch.__version__, torchvision.__version__, torch.cuda.is_available()
import os

import l5kit

import torch

import zarr

import pandas as pd

import numpy as np

from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.dataset import AgentDataset, EgoDataset
os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"
dm = LocalDataManager()
%%time

sample_dataset = ChunkedDataset(dm.require('scenes/sample.zarr'))

sample_dataset.open()

print(sample_dataset)
%%time

train_dataset = ChunkedDataset(dm.require('scenes/train.zarr'))

train_dataset.open()

print(train_dataset)
%%time

val_dataset = ChunkedDataset(dm.require('scenes/validate.zarr'))

val_dataset.open()

print(val_dataset)
%%time

test_dataset = ChunkedDataset(dm.require('scenes/test.zarr'))

test_dataset.open()

print(test_dataset)
del train_dataset

del val_dataset
sample_dataset.scenes
sample_dataset.frames
sample_dataset.tl_faces
sample_dataset.agents
agents_df = pd.DataFrame.from_records(sample_dataset.agents, columns = ['centroid', 'extent', 'yaw', 'velocity', 'track_id', 'label_probabilities'])

agents_df
agents_df.track_id.value_counts()
from l5kit.data import PERCEPTION_LABELS



agents_df['label_probabilities'].map(np.argmax).map(lambda i: PERCEPTION_LABELS[i]).value_counts()
agents_df['label_probabilities'].map(np.max).map(lambda p: int(p * 10) / 10.).value_counts()
scene_df = pd.DataFrame.from_records(sample_dataset.scenes, columns=('frame_index_interval', 'host', 'start_time', 'end_time'))

scene_df
scene_df['host'].value_counts()
frames_df = pd.DataFrame.from_records(sample_dataset.frames, columns = ['timestamp', 'agent_index_interval', 'traffic_light_faces_index_interval', 'ego_translation', 'ego_rotation'])

frames_df
tl_faces_df = pd.DataFrame.from_records(sample_dataset.tl_faces, columns = ['face_id', 'traffic_light_id', 'traffic_light_face_status'])

tl_faces_df
del sample_dataset
test_dataset.scenes
test_dataset.frames
test_dataset.agents
test_dataset.tl_faces
scene_df = pd.DataFrame.from_records(test_dataset.scenes, columns=('frame_index_interval', 'host', 'start_time', 'end_time'))

scene_df
agents_df = pd.DataFrame.from_records(zarr.array(test_dataset.agents[:50000]), 

                                      columns = ['centroid', 'extent', 'yaw', 'velocity', 'track_id', 'label_probabilities'])

agents_df
agents_df.track_id.value_counts()
frames_df = pd.DataFrame.from_records(zarr.array(test_dataset.frames[:10000]), columns = ['timestamp', 'agent_index_interval', 'traffic_light_faces_index_interval', 'ego_translation', 'ego_rotation'])

frames_df
CONFIG_DATA = {

    "format_version": 4,

    "model_params": {

        "model_architecture": "resnet50",

        # max is 99, but set to 101 never the less

        "history_num_frames": 101,

        "history_step_size": 1,

        "history_delta_time": 0.1,

        "future_num_frames": 50,

        "future_step_size": 1,

        "future_delta_time": 0.1,

    },

    "raster_params": {

        "raster_size": [256, 256],

        "pixel_size": [0.5, 0.5],

        "ego_center": [0.25, 0.5],

        "map_type": "py_semantic",

        "satellite_map_key": "aerial_map/aerial_map.png",

        "semantic_map_key": "semantic_map/semantic_map.pb",

        "dataset_meta_key": "meta.json",

        "filter_agents_threshold": 0.5,

        "disable_traffic_light_faces": False,

    },

    "test_dataloader": {

        "key": "scenes/test.zarr",

        "batch_size": 16,

        "shuffle": False,

        "num_workers": 4,

    },

}
test_mask = np.load(f"../input/lyft-motion-prediction-autonomous-vehicles/scenes/mask.npz")["arr_0"]
from l5kit.rasterization import build_rasterizer



rast = build_rasterizer(CONFIG_DATA, dm)
agent_dataset = AgentDataset(

    CONFIG_DATA, test_dataset, rast, agents_mask=test_mask

)
len(agent_dataset)
from tqdm.notebook import tqdm

from itertools import islice



items = []

track_ids = []

for i in tqdm(islice(agent_dataset, 20)):

    track_ids.append(i['track_id'])

    items.append(i)
len(track_ids), len(set(track_ids))
items[0].keys()
items[0]['track_id']
[len(item['history_availabilities']) for item in items]
items[0]['history_positions']
items[0]['history_yaws']