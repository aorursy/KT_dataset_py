!pip install git+https://github.com/rwightman/pytorch-image-models.git
from timm.models import vision_transformer
import torch
import l5kit, os
import torch.nn as nn
import numpy as np
import warnings;warnings.filterwarnings("ignore")
from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from tqdm import tqdm
from l5kit.geometry import transform_points
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable
import torch.optim as optim
import torch.nn.functional as F
from l5kit.evaluation import write_pred_csv
from l5kit.geometry import transform_points
os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"
model = vision_transformer.vit_small_resnet50d_s3_224(pretrained=True)
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
    }

}
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset
dm = LocalDataManager()
test_config = cfg["test_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
test_chunked = ChunkedDataset(dm.require(test_config["key"])).open()
test_mask = np.load(f"../input/lyft-motion-prediction-autonomous-vehicles/scenes/mask.npz")["arr_0"]
test_dataset = AgentDataset(cfg, test_chunked, rasterizer, agents_mask=test_mask)
test_loader = torch.utils.data.DataLoader(test_dataset,
                              shuffle=test_config["shuffle"],
                              batch_size=test_config["batch_size"],
                              num_workers=test_config["num_workers"])
class LyftVIT(nn.Module):
    
    def __init__(self, vit: nn.Module):
        super().__init__()
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        self.vit = vit
        num_targets = 2 * cfg["model_params"]["future_num_frames"]
        self.future_len = cfg["model_params"]["future_num_frames"]
        self.vit.patch_embed.backbone.conv1[0] = nn.Conv1d(
            num_in_channels,
            32,
            kernel_size=self.vit.patch_embed.backbone.conv1[0].kernel_size,
            stride=self.vit.patch_embed.backbone.conv1[0].stride,
            padding=self.vit.patch_embed.backbone.conv1[0].padding,
            bias=False,
        )
        
        
        self.num_preds = num_targets * 3
        self.num_modes = 3
        
        self.logit = nn.Linear(1000, out_features=self.num_preds + self.num_modes)
        
    def forward(self, x):
        x = self.vit(x)
        x = torch.flatten(x, 1)
        x = self.logit(x)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences
    
model = LyftVIT(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model initialized.")

model.vit.patch_embed.backbone.conv1[0] = nn.Conv2d(
            5,
            32,
            kernel_size=model.vit.patch_embed.backbone.conv1[0].kernel_size,
            stride=model.vit.patch_embed.backbone.conv1[0].stride,
            padding=model.vit.patch_embed.backbone.conv1[0].padding,
            bias=False,
        )
model.load_state_dict(torch.load('../input/lyft-vision-transformer-training/predictor.pt'))
model.vit.patch_embed.backbone.conv1[0] = nn.Conv2d(
            25,
            32,
            kernel_size=model.vit.patch_embed.backbone.conv1[0].kernel_size,
            stride=model.vit.patch_embed.backbone.conv1[0].stride,
            padding=model.vit.patch_embed.backbone.conv1[0].padding,
            bias=False,
        ).to(device)

# this is a bit hacky, kinda imperfect.
# the main issue is to transfer the weights with 25-channel input images
model.eval()

future_coords_offsets_pd = []
timestamps = []
agent_ids = []
confs = []

with torch.no_grad():
    dataiter = tqdm(test_loader)
    
    for data in dataiter:

        inputs = data["image"].to(device)
        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
        targets = data["target_positions"].to(device)
        
        outputs, conf = model(inputs)
        preds = outputs.cpu().numpy()
        conf = conf.cpu().numpy()
        world_from_agents = data["world_from_agent"].numpy()
        centroids = data["centroid"].numpy()
        coords_offset = []
        
        # convert into world coordinates and compute offsets
        for idx in range(len(preds)):
            for mode in range(3):
                preds[idx, mode, :, :] = transform_points(preds[idx, mode, :, :], world_from_agents[idx]) - centroids[idx][:2]
    
        future_coords_offsets_pd.append(preds.copy())
        confs.append(conf.copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy()) 
write_pred_csv('submission.csv',
               timestamps=np.concatenate(timestamps),
               track_ids=np.concatenate(agent_ids),
               coords=np.concatenate(future_coords_offsets_pd),
              confs=np.concatenate(confs))