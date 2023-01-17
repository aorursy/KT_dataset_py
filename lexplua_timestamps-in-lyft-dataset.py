# Installing l5kit offline
!pip install --no-index -f ../input/kaggle-l5kit pip==20.2.2 >/dev/nul
!pip install --no-index -f ../input/kaggle-l5kit -U l5kit > /dev/nul
import pandas as pd
import numpy as np
import os
import zarr
from l5kit.data import LocalDataManager, ChunkedDataset, get_frames_slice_from_scenes
from l5kit.dataset import AgentDataset
class NoImageAgentDataset(AgentDataset):
    '''Copy-pasted get_frame with small tweak to return empty matrix instead of image (to not use rasterizer)
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_frame(self, scene_index: int, state_index: int, track_id = None) -> dict:
        """
        A utility function to get the rasterisation and trajectory target for a given agent in a given frame

        Args:
            scene_index (int): the index of the scene in the zarr
            state_index (int): a relative frame index in the scene
            track_id (Optional[int]): the agent to rasterize or None for the AV
        Returns:
            dict: the rasterised image, the target trajectory (position and yaw) along with their availability,
            the 2D matrix to center that agent, the agent track (-1 if ego) and the timestamp

        """
        frames = self.dataset.frames[get_frames_slice_from_scenes(self.dataset.scenes[scene_index])]
        data = self.sample_function(state_index, frames, self.dataset.agents, self.dataset.tl_faces, track_id)

        target_positions = np.array(data["target_positions"], dtype=np.float32)
        target_yaws = np.array(data["target_yaws"], dtype=np.float32)

        history_positions = np.array(data["history_positions"], dtype=np.float32)
        history_yaws = np.array(data["history_yaws"], dtype=np.float32)

        timestamp = frames[state_index]["timestamp"]
        track_id = np.int64(-1 if track_id is None else track_id)  # always a number to avoid crashing torch

        return {
            "image": np.zeros(2),
            "target_positions": target_positions,
            "target_yaws": target_yaws,
            "target_availabilities": data["target_availabilities"],
            "history_positions": history_positions,
            "history_yaws": history_yaws,
            "history_availabilities": data["history_availabilities"],
            "world_to_image": data["world_to_image"],
            "track_id": track_id,
            "timestamp": timestamp,
            "centroid": data["centroid"],
            "yaw": data["yaw"],
            "extent": data["extent"],
        }
    @staticmethod
    def from_agent_ds(ds):
        return NoImageAgentDataset(
            ds.cfg,
            ds.dataset,
            ds.rasterizer,
            ds.perturbation,
            ds.agents_mask
        )



cfg = {
     'model_params': {
         'model_architecture': 'resnet50',
         'history_num_frames': 0, 
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
        'satellite_map_key': 
        'aerial_map/aerial_map.png', 
        'semantic_map_key': 'semantic_map/semantic_map.pb', 
        'dataset_meta_key': 'meta.json', 
        'filter_agents_threshold': 0.5
    }, 
    'sample_data_loader': {
        'key': 'scenes/sample.zarr', 
        'batch_size': 12, 
        'shuffle': False, 
        'num_workers': 16
    }
}
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"
dm = LocalDataManager()
dataset_path = dm.require(cfg['sample_data_loader']['key'])
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()


dataset = NoImageAgentDataset(cfg, zarr_dataset, None)
SCENE_IDX = 74
agents = dataset.from_agent_ds(dataset.get_scene_dataset(SCENE_IDX))
df = pd.DataFrame(iter(agents))
scene_record = zarr_dataset.scenes[SCENE_IDX]
display(scene_record['start_time'] - df['timestamp'].min(), scene_record['end_time'] - df['timestamp'].max())
