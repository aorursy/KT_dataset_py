!pip install --upgrade pip

!pip install pymap3d==2.1.0

!pip install -U l5kit
import l5kit, os

from l5kit.rasterization import build_rasterizer

from l5kit.configs import load_config_data

from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR

from l5kit.geometry import transform_points

from tqdm import tqdm

from collections import Counter

from l5kit.data import PERCEPTION_LABELS

from prettytable import PrettyTable

# set env variable for data

os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"

# get config

cfg = load_config_data("../input/lyft-config-files/visualisation_config.yaml")
from l5kit.data import ChunkedDataset, LocalDataManager

from l5kit.dataset import EgoDataset, AgentDataset

dm = LocalDataManager()

dataset_path = dm.require(cfg["val_data_loader"]["key"])

zarr_dataset = ChunkedDataset(dataset_path)

zarr_dataset.open()

print(zarr_dataset)
import numpy as np

from IPython.display import display, clear_output

import PIL

 

cfg["raster_params"]["map_type"] = "py_semantic"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, zarr_dataset, rast)

scene_idx = 2

indexes = dataset.get_scene_indices(scene_idx)

images = []



for idx in indexes:

    

    data = dataset[idx]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

    clear_output(wait=True)

    display(PIL.Image.fromarray(im[::-1]))
import numpy as np

from IPython.display import display, clear_output

import PIL

 

cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, zarr_dataset, rast)

scene_idx = 2

indexes = dataset.get_scene_indices(scene_idx)

images = []



for idx in indexes:

    

    data = dataset[idx]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

    clear_output(wait=True)

    display(PIL.Image.fromarray(im[::-1]))
from IPython.display import display, clear_output

from IPython.display import HTML



import PIL

import matplotlib.pyplot as plt

from matplotlib import animation, rc

def animate_solution(images):



    def animate(i):

        im.set_data(images[i])

 

    fig, ax = plt.subplots()

    im = ax.imshow(images[0])

    

    return animation.FuncAnimation(fig, animate, frames=len(images), interval=60)

cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, zarr_dataset, rast)

scene_idx = 34

indexes = dataset.get_scene_indices(scene_idx)

images = []



for idx in indexes:

    

    data = dataset[idx]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

    clear_output(wait=True)

    images.append(PIL.Image.fromarray(im[::-1]))

anim = animate_solution(images)

HTML(anim.to_jshtml())
from IPython.display import display, clear_output

import PIL

 

cfg["raster_params"]["map_type"] = "py_semantic"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, zarr_dataset, rast)

scene_idx = 34

indexes = dataset.get_scene_indices(scene_idx)

images = []



for idx in indexes:

    

    data = dataset[idx]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

    clear_output(wait=True)

    images.append(PIL.Image.fromarray(im[::-1]))

    

anim = animate_solution(images)

HTML(anim.to_jshtml())
import numpy as np

from IPython.display import display, clear_output

import PIL

 

cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset = AgentDataset(cfg, zarr_dataset, rast)

scene_idx = 2

indexes = dataset.get_scene_indices(scene_idx)

images = []



for idx in indexes:

    

    data = dataset[idx]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

    clear_output(wait=True)

    display(PIL.Image.fromarray(im[::-1]))
import numpy as np

from IPython.display import display, clear_output

import PIL

 

cfg["raster_params"]["map_type"] = "py_semantic"

rast = build_rasterizer(cfg, dm)

dataset = AgentDataset(cfg, zarr_dataset, rast)

scene_idx = 2

indexes = dataset.get_scene_indices(scene_idx)

images = []



for idx in indexes:

    

    data = dataset[idx]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

    clear_output(wait=True)

    display(PIL.Image.fromarray(im[::-1]))
from l5kit.data.map_api import MapAPI

from l5kit.rasterization.rasterizer_builder import _load_metadata



semantic_map_filepath = dm.require(cfg["raster_params"]["semantic_map_key"])

dataset_meta = _load_metadata(cfg["raster_params"]["dataset_meta_key"], dm)

world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)



map_api = MapAPI(semantic_map_filepath, world_to_ecef)

MAP_LAYERS = ["junction", "node", "segment", "lane"]





def element_of_type(elem, layer_name):

    return elem.element.HasField(layer_name)





def get_elements_from_layer(map_api, layer_name):

    return [elem for elem in map_api.elements if element_of_type(elem, layer_name)]





class MapRenderer:

    

    def __init__(self, map_api):

        self._color_map = dict(drivable_area='#a6cee3',

                               road_segment='#1f78b4',

                               road_block='#b2df8a',

                               lane='#474747')

        self._map_api = map_api

    

    def render_layer(self, layer_name):

        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_axes([0, 0, 1, 1])

        

    def render_lanes(self):

        all_lanes = get_elements_from_layer(self._map_api, "lane")

        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_axes([0, 0, 1, 1])

        for lane in all_lanes:

            self.render_lane(ax, lane)

        return fig, ax

        

    def render_lane(self, ax, lane):

        coords = self._map_api.get_lane_coords(MapAPI.id_as_str(lane.id))

        self.render_boundary(ax, coords["xyz_left"])

        self.render_boundary(ax, coords["xyz_right"])

        

    def render_boundary(self, ax, boundary):

        xs = boundary[:, 0]

        ys = boundary[:, 1] 

        ax.plot(xs, ys, color=self._color_map["lane"], label="lane")

        

        

renderer = MapRenderer(map_api)

fig, ax = renderer.render_lanes()
from typing import Dict

!pip install pytorch-lightning



from tempfile import gettempdir

import matplotlib.pyplot as plt

import numpy as np

import torch

from torch import nn, optim

from torch.utils.data import DataLoader

from torchvision.models.resnet import resnet18

from tqdm import tqdm

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

import pytorch_lightning as pl

import os

cfg = load_config_data('../input/lyft-config-files/agent_motion_config.yaml')

class Mod(torch.nn.Module):

    def __init__(self, cfg: Dict):

        super(Mod, self).__init__()

        self.backbone = resnet18(pretrained=False)

        

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2

        num_in_channels = 3 + num_history_channels



        self.backbone.conv1 = nn.Conv2d(

            num_in_channels,

            self.backbone.conv1.out_channels,

            kernel_size=self.backbone.conv1.kernel_size,

            stride=self.backbone.conv1.stride,

            padding=self.backbone.conv1.padding,

            bias=False,

        )

        

        # This is 512 for resnet18 and resnet34;

        # And it is 2048 for the other resnets

        backbone_out_features = 512



        # X, Y coords for the future positions (output shape: Bx50x2)

        num_targets = 2 * cfg["model_params"]["future_num_frames"]



        # You can add more layers here.

        self.head = nn.Sequential(

            # nn.Dropout(0.2),

            nn.Linear(in_features=backbone_out_features, out_features=4096),

        )



        self.logit = nn.Linear(4096, out_features=num_targets)

    def forward(self):

        x = self.backbone.conv1(x)

        x = self.backbone.bn1(x)

        x = self.backbone.relu(x)

        x = self.backbone.maxpool(x)



        x = self.backbone.layer1(x)

        x = self.backbone.layer2(x)

        x = self.backbone.layer3(x)

        x = self.backbone.layer4(x)



        x = self.backbone.avgpool(x)

        x = torch.flatten(x, 1)

        

        x = self.head(x)

        x = self.logit(x)

        

        return x        



def forward(data, model, device, criterion):

    inputs = data["image"].to(device)

    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)

    targets = data["target_positions"].to(device)

    # Forward pass

    outputs = model(inputs).reshape(targets.shape)

    loss = criterion(outputs, targets)

    # not all the output steps are valid, but we can filter them out from the loss using availabilities

    loss = loss * target_availabilities

    loss = loss.mean()

    return loss, outputs



class LightningLyft(pl.LightningModule):

    def __init__(self, model):

        super(LightningLyft, self).__init__()

        self.model = model

        

    def forward(self, x, *args, **kwargs):

        return self.model(x)

    

    def prepare_train_data(self):

        train_cfg = cfg["train_data_loader"]

        rasterizer = build_rasterizer(cfg, dm)

        train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()

        train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

        train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 

                             num_workers=train_cfg["num_workers"])

        return train_dataloader

            

    def training_step(self, batch, batch_idx):

        tr_it = iter(train_dataloader)

        progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))

        losses_train = []

        model = self.model

        for n in [0, 1, 2 , 3, 4]:

            try:

                data = next(tr_it)

            except StopIteration:

                tr_it = iter(train_dataloader)

                data = next(tr_it)

            model.train()

            torch.set_grad_enabled(True)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            criterion = nn.MSELoss(reduction="none")

            loss, _ = forward(data, model, device, criterion)



            # Backward pass

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



            losses_train.append(loss.item())

            print(f"LOSS FOR EPOCH {n}: {loss.item()}")

            

    def configure_optimizers(self):

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        return optimizer
# ===== INIT DATASET

train_cfg = cfg["train_data_loader"]

rasterizer = build_rasterizer(cfg, dm)

train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()

train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 

                             num_workers=train_cfg["num_workers"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Mod(cfg).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.MSELoss(reduction="none")
# ==== TRAIN LOOP

res = []

tr_it = iter(train_dataloader)

model = LightningLyft(build_model(cfg))

progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))

losses_train = []

for _ in progress_bar:

    try:

        data = next(tr_it)

    except StopIteration:

        tr_it = iter(train_dataloader)

        data = next(tr_it)

    model.train()

    torch.set_grad_enabled(True)

    loss, _ = forward(data, model, device, criterion)

    res.append(_)

    # Backward pass

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()



    losses_train.append(loss.item())

    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")