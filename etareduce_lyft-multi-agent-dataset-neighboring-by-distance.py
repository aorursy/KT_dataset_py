!pip install -q pip==20.2.3

!pip uninstall -y typing

!pip install -q l5kit==1.1 pytorch-lightning==0.10.0
import bisect

import os

from copy import deepcopy

from operator import itemgetter

from typing import Any, Dict, List, Optional, Tuple



import numpy as np

import pytorch_lightning as pl

from l5kit.data import ChunkedDataset, LocalDataManager

from l5kit.dataset import AgentDataset

from l5kit.rasterization import StubRasterizer, build_rasterizer

from torch.utils.data import DataLoader, Dataset, Subset



is_kaggle = os.path.isdir("/kaggle")





data_root = (

    "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"

    if is_kaggle

    else "lyft-motion-prediction-autonomous-vehicles"

)





CONFIG_DATA = {

    "format_version": 4,

    "model_params": {

        "model_architecture": "resnet34",

        "history_num_frames": 10,

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

    "train_dataloader": {

        "key": "scenes/sample.zarr",

        "batch_size": 24,

        "shuffle": True,

        "num_workers": 0,

    },

    "val_dataloader": {

        "key": "scenes/validate.zarr",

        "batch_size": 24,

        "shuffle": False,

        "num_workers": 4,

    },

    "test_dataloader": {

        "key": "scenes/test.zarr",

        "batch_size": 24,

        "shuffle": False,

        "num_workers": 4,

    },

    "train_params": {

        "max_num_steps": 400,

        "eval_every_n_steps": 50,

    },

}
class MultiAgentDataset(Dataset):

    def __init__(

        self,

        rast_only_agent_dataset: AgentDataset,

        history_agent_dataset: AgentDataset,

        num_neighbors: int = 10,

    ):

        super().__init__()

        self.rast_only_agent_dataset = rast_only_agent_dataset

        self.history_agent_dataset = history_agent_dataset

        self.num_neighbors = num_neighbors



    def __len__(self) -> int:

        return len(self.rast_only_agent_dataset)



    def get_others_dict(

        self, index: int, ego_dict: Dict[str, Any]

    ) -> Tuple[List[Dict[str, Any]], int]:

        agent_index = self.rast_only_agent_dataset.agents_indices[index]

        frame_index = bisect.bisect_right(

            self.rast_only_agent_dataset.cumulative_sizes_agents, agent_index

        )

        frame_indices = self.rast_only_agent_dataset.get_frame_indices(frame_index)

        assert len(frame_indices) >= 1, frame_indices

        frame_indices = frame_indices[frame_indices != index]



        others_dict = []

        # The centroid of the AV in the current frame in world reference system. Unit is meters

        for idx, agent in zip(  # type: ignore

            frame_indices,

            Subset(self.history_agent_dataset, frame_indices),

        ):

            agent["dataset_idx"] = idx

            agent["dist_to_ego"] = np.linalg.norm(

                agent["centroid"] - ego_dict["centroid"], ord=2

            )

            # TODO in future we can convert history positions via agent + ego transformation matrix

            # TODO and get the normalized version

            del agent["image"]

            others_dict.append(agent)



        others_dict = sorted(others_dict, key=itemgetter("dist_to_ego"))

        others_dict = others_dict[: self.num_neighbors]

        others_len = len(others_dict)



        # have to pad because torch has no ragged tensor

        # https://github.com/pytorch/pytorch/issues/25032

        length_to_pad = self.num_neighbors - others_len

        pad_item = deepcopy(ego_dict)

        pad_item["dataset_idx"] = index

        pad_item["dist_to_ego"] = np.nan  # set to nan so you don't by chance use this

        del pad_item["image"]

        return (others_dict + [pad_item] * length_to_pad, others_len)



    def __getitem__(self, index: int) -> Dict[str, Any]:

        rast_dict = self.rast_only_agent_dataset[index]

        ego_dict = self.history_agent_dataset[index]

        others_dict, others_len = self.get_others_dict(index, ego_dict)

        ego_dict["image"] = rast_dict["image"]

        return {

            "ego_dict": ego_dict,

            "others_dict": others_dict,

            "others_len": others_len,

        }
class LyftAgentDataModule(pl.LightningDataModule):

    def __init__(self, cfg: Dict = CONFIG_DATA, data_root: str = data_root):

        super().__init__()

        self.cfg = cfg

        self.dm = LocalDataManager(data_root)

        self.rast = build_rasterizer(self.cfg, self.dm)



    def chunked_dataset(self, key: str):

        dl_cfg = self.cfg[key]

        dataset_path = self.dm.require(dl_cfg["key"])

        zarr_dataset = ChunkedDataset(dataset_path)

        zarr_dataset.open()

        return zarr_dataset



    def get_dataloader_by_key(

        self, key: str, mask: Optional[np.ndarray] = None

    ) -> DataLoader:

        dl_cfg = self.cfg[key]

        zarr_dataset = self.chunked_dataset(key)

        agent_dataset = AgentDataset(

            self.cfg, zarr_dataset, self.rast, agents_mask=mask

        )

        return DataLoader(

            agent_dataset,

            shuffle=dl_cfg["shuffle"],

            batch_size=dl_cfg["batch_size"],

            num_workers=dl_cfg["num_workers"],

            pin_memory=True,

        )



    def train_dataloader(self):

        key = "train_dataloader"

        return self.get_dataloader_by_key(key)



    def val_dataloader(self):

        key = "val_dataloader"

        return self.get_dataloader_by_key(key)



    def test_dataloader(self):

        key = "test_dataloader"

        test_mask = np.load(f"{data_root}/scenes/mask.npz")["arr_0"]

        return self.get_dataloader_by_key(key, mask=test_mask)
class MultiAgentDataModule(LyftAgentDataModule):

    def __init__(self, cfg: Dict = CONFIG_DATA, data_root: str = data_root) -> None:

        super().__init__(cfg=cfg, data_root=data_root)

        stub_cfg = deepcopy(self.cfg)



        # this can be removed once l5kit supporting passing None as rasterizer

        # https://github.com/lyft/l5kit/pull/176

        stub_cfg["raster_params"]["map_type"] = "stub_debug"

        self.stub_rast = build_rasterizer(stub_cfg, self.dm)

        assert isinstance(self.stub_rast, StubRasterizer)



    def get_dataloader_by_key(

        self, key: str, mask: Optional[np.ndarray] = None

    ) -> DataLoader:

        dl_cfg = self.cfg[key]

        zarr_dataset = self.chunked_dataset(key)

        # for the rast only agent dataset, we'll disable rasterization for history frames, so the

        # channel size will only be 3 + 2 (for current frame)

        no_history_cfg = deepcopy(self.cfg)

        no_history_cfg["model_params"]["history_num_frames"] = 0

        rast_only_agent_dataset = AgentDataset(

            no_history_cfg, zarr_dataset, self.rast, agents_mask=mask

        )

        history_agent_dataset = AgentDataset(

            self.cfg, zarr_dataset, self.stub_rast, agents_mask=mask

        )

        return DataLoader(

            MultiAgentDataset(

                rast_only_agent_dataset=rast_only_agent_dataset,

                history_agent_dataset=history_agent_dataset,

            ),

            shuffle=dl_cfg["shuffle"],

            batch_size=dl_cfg["batch_size"],

            num_workers=dl_cfg["num_workers"],

            pin_memory=True,

        )
datamodule = MultiAgentDataModule()
from pprint import pprint

for item in datamodule.train_dataloader():

    pprint(item.keys())

    print('ego_dict keys')

    pprint(item['ego_dict'].keys())

    pprint(len(item['others_dict']))

    pprint(item['others_dict'][0].keys())

    pprint(item['others_len'])

    break