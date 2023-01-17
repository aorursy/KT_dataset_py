import numpy as np

import os

import torch



from torch import nn, optim

from torch.utils.data import DataLoader

from torchvision.models.resnet import resnet18

from tqdm import tqdm

from typing import Dict



from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.dataset import AgentDataset, EgoDataset

from l5kit.rasterization import build_rasterizer

DIR_INPUT = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"



SINGLE_MODE_SUBMISSION = f"{DIR_INPUT}/single_mode_sample_submission.csv"

MULTI_MODE_SUBMISSION = f"{DIR_INPUT}/multi_mode_sample_submission.csv"



DEBUG = True
cfg = {

    'format_version': 4,

    'model_params': {

        'model_architecture': 'resnet50',

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

    

    'train_data_loader': {

        'key': 'scenes/train.zarr',

        'batch_size': 12,

        'shuffle': True,

        'num_workers': 4

    },

    

    'train_params': {

        'max_num_steps': 100 if DEBUG else 10000,

        'checkpoint_every_n_steps': 5000,

        

        # 'eval_every_n_steps': -1

    }

}
# set env variable for data

os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT

dm = LocalDataManager(None)
# ===== INIT DATASET

train_cfg = cfg["train_data_loader"]



# Rasterizer

rasterizer = build_rasterizer(cfg, dm)



# Train dataset/dataloader

train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()

train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

train_dataloader = DataLoader(train_dataset,

                              shuffle=train_cfg["shuffle"],

                              batch_size=train_cfg["batch_size"],

                              num_workers=train_cfg["num_workers"])



print(train_dataset)
class LyftModel(nn.Module):

    

    def __init__(self, cfg: Dict):

        super().__init__()

        

        self.backbone = resnet18(pretrained=True, progress=True)

        

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

        

    def forward(self, x):

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

# ==== INIT MODEL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model = LyftModel(cfg)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)



# Later we have to filter the invalid steps.

criterion = nn.MSELoss(reduction="none")

device
# ==== TRAIN LOOP

tr_it = iter(train_dataloader)



progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))

losses_train = []



for itr in progress_bar:



    try:

        data = next(tr_it)

    except StopIteration:

        tr_it = iter(train_dataloader)

        data = next(tr_it)



    model.train()

    torch.set_grad_enabled(True)

    

    # Forward pass

    inputs = data["image"].to(device)

    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)

    targets = data["target_positions"].to(device)

    

    outputs = model(inputs).reshape(targets.shape)

    loss = criterion(outputs, targets)



    # not all the output steps are valid, but we can filter them out from the loss using availabilities

    loss = loss * target_availabilities

    loss = loss.mean()



    # Backward pass

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()



    losses_train.append(loss.item())



    if (itr+1) % cfg['train_params']['checkpoint_every_n_steps'] == 0 and not DEBUG:

        torch.save(model.state_dict(), f'model_state_{itr}.pth')

    

    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train[-100:])}")
if not DEBUG:

    torch.save(model.state_dict(), f'model_state_last.pth')
model.eval()



future_coords_offsets_pd = []

timestamps = []

agent_ids = []



with torch.no_grad():

    dataiter = tqdm(iter(test_dataloader))

    

    for data in dataiter:



        inputs = data["image"].to(device)

        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)

        targets = data["target_positions"].to(device)



        outputs = model(inputs).reshape(targets.shape)

        

        future_coords_offsets_pd.append(outputs.cpu().numpy().copy())

        timestamps.append(data["timestamp"].numpy().copy())

        agent_ids.append(data["track_id"].numpy().copy())

write_pred_csv('submission.csv',

               timestamps=np.concatenate(timestamps),

               track_ids=np.concatenate(agent_ids),

               coords=np.concatenate(future_coords_offsets_pd))