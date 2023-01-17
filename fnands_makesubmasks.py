!pip install --no-index -f ../input/kaggle-l5kit pip==20.2.2 >/dev/nul

!pip install --no-index -f ../input/kaggle-l5kit -U l5kit > /dev/nul

from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.dataset import AgentDataset

from l5kit.rasterization import build_rasterizer

from l5kit.evaluation import write_pred_csv

from l5kit.data.filter import get_agents_slice_from_frames

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader



import os
os.environ["L5KIT_DATA_FOLDER"] = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"

# local data manager

dm = LocalDataManager()

# set dataset path

dataset_path = dm.require('scenes/train.zarr')

# load the dataset; this is a zarr format, chunked dataset

chunked_dataset = ChunkedDataset(dataset_path)

# open the dataset

chunked_dataset.open()
cfg = {

    'format_version': 4,

    'model_params': {

        'history_num_frames': 99,

        'history_step_size': 1,

        'history_delta_time': 0.1,

        'future_num_frames': 50,

        'future_step_size': 1,

        'future_delta_time': 0.1

    },

    

    'raster_params': {

        'raster_size': [1, 1],

        'pixel_size': [0.5, 0.5],

        'ego_center': [0.5, 0.5],

        'map_type': 'box_debug',

        'satellite_map_key': 'aerial_map/aerial_map.png',

        'semantic_map_key': 'semantic_map/semantic_map.pb',

        'dataset_meta_key': 'meta.json',

        'filter_agents_threshold': 0.5,

        'disable_traffic_light_faces' : False



    },

    

    'sample_data_loader': {

        'key': 'scenes/sample.zarr',

        'batch_size': 4,

        'shuffle': False,

        'num_workers': 8

    }

}
n_frames = len(chunked_dataset.frames)

frame_mask = np.zeros((n_frames,))
interval = 10

start_frame = 100

end_frame = 200





for scene in chunked_dataset.scenes:

    f1, _ = scene['frame_index_interval']

    for frame_no in np.arange(f1 + start_frame, f1 + end_frame + 1, interval):

        #ag_s = get_agents_slice_from_frames(chunked_dataset.frames[frame_no])

        #print(frame_no)

        frame_mask[frame_no] = 1

# Create the name of the oputput file

outfile = "frame_mask_" + str(start_frame) + "_" + str(end_frame) + "_" + str(interval)



# Save the mask

np.savez(outfile, frame_mask.astype(bool))