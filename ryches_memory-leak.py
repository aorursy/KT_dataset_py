import torch

# torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data import DataLoader

from tqdm.notebook import tqdm



from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.rasterization import build_rasterizer

from l5kit.dataset import AgentDataset



import os
DIR_INPUT = "../input/lyft-motion-prediction-autonomous-vehicles"
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

        'num_workers': 4

    },

    

    'train_data_loader': {

        'key': 'scenes/train.zarr',

        'batch_size': 32,

        'shuffle': True,

        'num_workers': 4

    },

    

    'val_data_loader': {

        'key': 'scenes/validate.zarr',

        'batch_size': 8,

        'shuffle': True,

        'num_workers': 4

    },

    'train_params':{

      'checkpoint_every_n_steps': 10000,

      'max_num_steps': 2000,

      'eval_every_n_steps': 10000

                    }



}
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT

dm = LocalDataManager(None)
rasterizer = build_rasterizer(cfg, dm)

train_zarr = ChunkedDataset(dm.require(cfg['train_data_loader']["key"])).open(cached=False)

train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
epochs = 4

train_dataloader = DataLoader(train_dataset,

                 shuffle=True,

                 batch_size=cfg['train_data_loader']['batch_size'],

                 num_workers=cfg['train_data_loader']['num_workers'])

for i in range(epochs):

    tr_it = iter(train_dataloader)

    progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))

    for _ in progress_bar:

        try:

            data = next(tr_it)

        except StopIteration:

            tr_it = iter(train_dataloader)

            data = next(tr_it)