from IPython.display import HTML

HTML('<center><iframe width="700" height="400" src="https://www.youtube.com/embed/tlThdr3O5Qo?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe></center>')
!pip install neptune-client segmentation_models_pytorch hydra-core kekas -U -q 

!pip install --target=/kaggle/working pymap3d==2.1.0 -q

!pip install --target=/kaggle/working strictyaml -q

!pip install --target=/kaggle/working protobuf==3.12.2 -q

!pip install --target=/kaggle/working transforms3d -q

!pip install --target=/kaggle/working zarr -q

!pip install --target=/kaggle/working ptable -q

!pip install --no-dependencies --target=/kaggle/working l5kit==1.1.0 --upgrade -q

!cp ../input/lyft-config-files/agent_motion_config.yaml config.yaml

import l5kit, os, albumentations as A

from l5kit.rasterization import build_rasterizer

from l5kit.configs import load_config_data

from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR

from tqdm import tqdm

from l5kit.geometry import transform_points

from collections import Counter

from l5kit.data import PERCEPTION_LABELS

from prettytable import PrettyTable

import matplotlib.pyplot as plt

# set env variable for data

os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"

# get config

MONITORING = True # set this to false if you want to fork and train

if MONITORING:

    import utilsforlyft as U

cfg = load_config_data("../input/lyft-config-files/visualisation_config.yaml")

import plotly.offline as py

import omegaconf

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import torch

import torch.nn.functional as F

import torch.nn as nn

from catalyst import dl, data

from catalyst.utils import metrics

from torch.utils.data import DataLoader

from catalyst.dl import utils, BatchOverfitCallback

from torch.optim.lr_scheduler import OneCycleLR

import segmentation_models_pytorch as smp

from catalyst.core.callbacks.early_stop import EarlyStoppingCallback

if MONITORING:

    from catalyst.contrib.dl.callbacks.neptune_logger import NeptuneLogger

    from catalyst.contrib.dl.callbacks import WandbLogger

    neptune_logger = NeptuneLogger(

                    api_token=U.TOKEN + '=',  

                    project_name="trigram19/"+U.NAME_PROJ,

                    offline_mode=False, 

                    name=U.NAME,

                    params={'epoch_nr': 5}, 

                    properties={'data_source': 'lyft'},  

                    tags=['resnet']

                    )



from IPython.display import display, clear_output, HTML

import PIL

import matplotlib.pyplot as plt

from matplotlib import animation as ani, rc

import numpy as np

import warnings; warnings.filterwarnings('ignore')

from l5kit.rasterization.rasterizer_builder import _load_metadata

from kekas import Keker, DataOwner, DataKek

from kekas.utils import DotDict

from kekas.transformations import Transformer, to_torch, normalize
def plot_image(map_type, ax, agent=False):

    cfg["raster_params"]["map_type"] = map_type

    rast = build_rasterizer(cfg, dm)

    if agent:

        dataset = AgentDataset(cfg, zarr_dataset, rast)

    else:

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

        draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, yaws=data["target_yaws"])

        clear_output(wait=True)

        ax.imshow(im[::-1])

                

def animate_solution(images):

    def animate(i):

        im.set_data(images[i]) 

    fig, ax = plt.subplots()

    im = ax.imshow(images[0])    

    return ani.FuncAnimation(fig, animate, frames=len(images), interval=60)



def animation(type_):

    cfg["raster_params"]["map_type"] = type_

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

        draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, yaws=data["target_yaws"])

        clear_output(wait=True)

        images.append(PIL.Image.fromarray(im[::-1]))

    anim = animate_solution(images)

    return HTML(anim.to_jshtml())
from l5kit.data import ChunkedDataset, LocalDataManager

from l5kit.dataset import EgoDataset, AgentDataset

dm = LocalDataManager()

dataset_path = dm.require(cfg["val_data_loader"]["key"]);rasterizer = build_rasterizer(cfg, dm)

zarr_dataset = ChunkedDataset(dataset_path)

train_dataset_a = AgentDataset(cfg, zarr_dataset, rasterizer)

zarr_dataset.open()

print(zarr_dataset)
fig, axes = plt.subplots(1, 1, figsize=(15, 10))

for i, key in enumerate(["py_semantic"]):

    plot_image(key, ax=axes)
fig, axes = plt.subplots(1, 1, figsize=(15, 10))

for i, key in enumerate(["py_satellite"]):

    plot_image(key, ax=axes)
animation("py_satellite")
animation("py_semantic")
animation("box_debug")
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
fig, axes = plt.subplots(1, 1, figsize=(15, 10))

for i, key in enumerate(["py_satellite"]):

    plot_image(key, ax=axes, agent=True)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for i, key in enumerate(['box_debug', 'stub_debug', 'py_semantic', 'py_satellite']):

    plot_image(key, ax=axes)
print("scenes", zarr_dataset.scenes)

print("scenes[0]", zarr_dataset.scenes[0])
import pandas as pd

scenes = zarr_dataset.scenes

scenes_df = pd.DataFrame(scenes)

scenes_df.columns = ["data"]; features = ['frame_index_interval', 'host', 'start_time', 'end_time']

for i, feature in enumerate(features):

    scenes_df[feature] = scenes_df['data'].apply(lambda x: x[i])

scenes_df.drop(columns=["data"],inplace=True)

print(f"scenes dataset: {scenes_df.shape}")

scenes_df.head()
agents = pd.read_csv('../input/lyft-motion-prediction-autonomous-vehicles-as-csv/agents_0_10019001_10019001.csv')

agents
import seaborn as sns

colormap = plt.cm.magma

cont_feats = ["centroid_x", "centroid_y", "extent_x", "extent_y", "extent_z", "yaw"]

plt.figure(figsize=(16,12));

plt.title('Pearson correlation of features', y=1.05, size=15);

sns.heatmap(agents[cont_feats].corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True);
import seaborn as sns

plot = sns.jointplot(x=agents['centroid_x'][:1000], y=agents['centroid_y'][:1000], kind='hexbin', color='blueviolet')

plot.set_axis_labels('center_x', 'center_y', fontsize=16)



plt.show()
fig = plt.figure(figsize=(15, 15));

sns.distplot(agents['extent_x'], color='steelblue');

sns.distplot(agents['extent_y'], color='purple');



plt.title("Distributions of Extents X and Y");
fig = plt.figure(figsize=(15, 15));

sns.distplot(agents['extent_z'], color='steelblue');



plt.title("Distributions of Extents z");
fig = plt.figure(figsize=(15, 15));

sns.distplot(agents['yaw'], color='steelblue');



plt.title("Distributions of Extents z");
frms = pd.read_csv("../input/lyft-motion-prediction-autonomous-vehicles-as-csv/frames_0_124167_124167.csv")

frms.head()
import seaborn as sns

colormap = plt.cm.magma

cont_feats = ["ego_rotation_xx", "ego_rotation_xy", "ego_rotation_xz", "ego_rotation_yx", "ego_rotation_yy", "ego_rotation_yz", "ego_rotation_zx", "ego_rotation_zy", "ego_rotation_zz"]

plt.figure(figsize=(16,12));

plt.title('Pearson correlation of features', y=1.05, size=15);

sns.heatmap(frms[cont_feats].corr(),linewidths=0.1,vmax=1.0, square=True, 

            cmap=colormap, linecolor='white', annot=True);

cfg2 = load_config_data("../input/lyft-config-files/agent_motion_config.yaml")

cfg2 = omegaconf.DictConfig(cfg2)

train_cfg = omegaconf.OmegaConf.to_container(cfg2.train_data_loader)

validation_cfg = omegaconf.OmegaConf.to_container(cfg2.val_data_loader)

# Rasterizer

rasterizer = build_rasterizer(cfg2, dm)
print(cfg2.pretty())
class LyftModel(torch.nn.Module):

    

    def __init__(self, cfg: omegaconf.dictconfig.DictConfig):

        super().__init__()

        self.backbone = smp.FPN(encoder_name="resnext50_32x4d", classes=1)

        

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2

        num_in_channels = 3 + num_history_channels

        

        self.backbone.encoder.conv1 = nn.Conv2d(

            num_in_channels,

             self.backbone.encoder.conv1.out_channels,

            kernel_size= self.backbone.encoder.conv1.kernel_size,

            stride= self.backbone.encoder.conv1.stride,

            padding= self.backbone.encoder.conv1.padding,

            bias=False,

        )

        

        backbone_out_features = 14

        

        num_targets = 2 * cfg["model_params"]["future_num_frames"]

        

        self.head = nn.Sequential(

            nn.Dropout(0.2),

            nn.Linear(in_features=14, out_features=4096),

        )

        

        self.backbone.segmentation_head = nn.Sequential(nn.Conv1d(56, 1, kernel_size=3, stride=2), nn.Dropout(0.2), nn.ReLU())

        

        self.logit = nn.Linear(4096, out_features=num_targets)

        self.logit_final = nn.Linear(128, 12)

        self.num_preds = num_targets * 3



    def forward(self, x):

        x = self.backbone.encoder.conv1(x)

        x = self.backbone.encoder.bn1(x)

        

        x = self.backbone.encoder.relu(x)

        x = self.backbone.encoder.maxpool(x)

        

        x = self.backbone.encoder.layer1(x)

        x = self.backbone.encoder.layer2(x)

        x = self.backbone.encoder.layer3(x)

        x = self.backbone.encoder.layer4(x)

        

        x = self.backbone.decoder.p5(x)

        x = self.backbone.decoder.seg_blocks[0](x)

        x = self.backbone.decoder.merge(x)

        

        x = self.backbone.segmentation_head(x)

        x = self.backbone.encoder.maxpool(x)

        

        x = torch.flatten(x, 1)

        x = self.head(x)

        x = self.logit(x)

        

        x = x.permute(1, 0)

        x = self.logit_final(x)

        return x
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LyftModel(cfg2)

model.to(device)

train_zarr = ChunkedDataset(dm.require(train_cfg['key'])).open()

train_dataset = AgentDataset(cfg2, train_zarr, rasterizer)

del train_cfg['key']

subset = torch.utils.data.Subset(train_dataset, range(0, 1100))

train_dataloader = DataLoader(subset,

                              **train_cfg)

val_zarr = ChunkedDataset(dm.require(validation_cfg['key'])).open()

del validation_cfg['key']



val_dataset = AgentDataset(cfg2, val_zarr, rasterizer)

subset = torch.utils.data.Subset(val_dataset, range(0, 50))

val_dataloader = DataLoader(subset,

                              **validation_cfg)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)



loaders = {

    "train": train_dataloader,

    "valid": val_dataloader

}



class LyftRunner(dl.SupervisedRunner):



    def predict_batch(self, batch):

        return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))



    def _handle_batch(self, batch):

        x, y = batch['image'], batch['target_positions']

        y_hat = self.model(x).view(y.shape)

        target_availabilities = batch["target_availabilities"].unsqueeze(-1)

        criterion = torch.nn.MSELoss(reduction="none")

        

        loss = criterion(y_hat, y)

        loss = loss * target_availabilities

        loss = loss.mean()

        self.batch_metrics.update(

            {"loss": loss}

        )

%%time

device = utils.get_device()

runner = LyftRunner(device=device, input_key="image", input_target_key="target_positions", output_key="logits")

if MONITORING:



    runner.train(

        model=model,

        optimizer=optimizer,

        loaders=loaders,

        logdir="../working",

        num_epochs=4,

        verbose=True,

        load_best_on_end=True,

        callbacks=[neptune_logger, BatchOverfitCallback(train=10, valid=0.5), 

                  EarlyStoppingCallback(

            patience=2,

            metric="loss",

            minimize=True,

        ), WandbLogger(project="dertaismus",name= 'Example')

                  ]

    )

else:

    runner.train(

        model=model,

        optimizer=optimizer,

        loaders=loaders,

        logdir="../working",

        num_epochs=4,

        verbose=True,

        load_best_on_end=True,

        callbacks=[BatchOverfitCallback(train=10, valid=0.5), 

                  EarlyStoppingCallback(

            patience=2,

            metric="loss",

            minimize=True,

        )

                  ]

    )

    

# train for more steps and loss will not be low

# or at least not as pathetic as the loss over here    
dataowner = DataOwner(train_dataloader, val_dataloader, None)

def step_fn(model: torch.nn.Module,

            batch: torch.Tensor) -> torch.Tensor:

    

    inp,t = batch["image"], batch["target_positions"]

    model.logit_final = nn.Linear(128, t.shape[0]).cuda()

    return model(inp).reshape(t.shape)
keker = Keker(model=model,

              dataowner=dataowner,

              criterion=torch.nn.MSELoss(),

              step_fn=step_fn,

              target_key="target_positions",

              opt=torch.optim.SGD,

              opt_params={"momentum": 0.99})
keker.unfreeze(model_attr="backbone")



layer_num = -1

keker.freeze_to(layer_num, model_attr="backbone")
keker.kek( lr=1e-3, 

            epochs=5,                  

            logdir='train_logs')



# loss computation is slightly faulty I'll try to fix it if possible

# issue is that we need to multiply loss by target availabilities like we did in Catalyst model

# however that does not work out of the box with kekas as far as I know, this is the main reason loss is going yoyo



keker.plot_kek('train_logs')