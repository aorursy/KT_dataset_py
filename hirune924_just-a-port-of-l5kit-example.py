!pip install --upgrade pip

!pip install pymap3d==2.1.0

!pip install -U l5kit
from typing import Dict



from tempfile import gettempdir

import matplotlib.pyplot as plt

import numpy as np

import torch

from torch import nn, optim

from torch.utils.data import DataLoader

from torchvision.models.resnet import resnet50

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



import os
!cp -r ../input/lyft-motion-prediction-autonomous-vehicles ../lyft-motion-prediction-autonomous-vehicles
# set env variable for data

os.environ["L5KIT_DATA_FOLDER"] = "../lyft-motion-prediction-autonomous-vehicles"

dm = LocalDataManager(None)

# get config

cfg = load_config_data("../input/lyft-config-files/agent_motion_config.yaml")

print(cfg)
def build_model(cfg: Dict) -> torch.nn.Module:

    # load pre-trained Conv2D model

    model = resnet50(pretrained=True)



    # change input channels number to match the rasterizer's output

    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2

    num_in_channels = 3 + num_history_channels

    model.conv1 = nn.Conv2d(

        num_in_channels,

        model.conv1.out_channels,

        kernel_size=model.conv1.kernel_size,

        stride=model.conv1.stride,

        padding=model.conv1.padding,

        bias=False,

    )

    # change output size to (X, Y) * number of future states

    num_targets = 2 * cfg["model_params"]["future_num_frames"]

    model.fc = nn.Linear(in_features=2048, out_features=num_targets)



    return model
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
# ===== INIT DATASET

train_cfg = cfg["train_data_loader"]

rasterizer = build_rasterizer(cfg, dm)

train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()

train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 

                             num_workers=train_cfg["num_workers"])

print(train_dataset)
# ==== INIT MODEL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = build_model(cfg).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.MSELoss(reduction="none")
# ==== TRAIN LOOP

tr_it = iter(train_dataloader)

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



    # Backward pass

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()



    losses_train.append(loss.item())

    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")
# ===== GENERATE AND LOAD CHOPPED DATASET

num_frames_to_chop = 100

eval_cfg = cfg["val_data_loader"]

eval_base_path = create_chopped_dataset(dm.require(eval_cfg["key"]), cfg["raster_params"]["filter_agents_threshold"], 

                              num_frames_to_chop, cfg["model_params"]["future_num_frames"], MIN_FUTURE_STEPS)
eval_zarr_path = str(Path(eval_base_path) / Path(dm.require(eval_cfg["key"])).name)

eval_mask_path = str(Path(eval_base_path) / "mask.npz")

eval_gt_path = str(Path(eval_base_path) / "gt.csv")



eval_zarr = ChunkedDataset(eval_zarr_path).open()

eval_mask = np.load(eval_mask_path)["arr_0"]

# ===== INIT DATASET AND LOAD MASK

eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer, agents_mask=eval_mask)

eval_dataloader = DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"], 

                             num_workers=eval_cfg["num_workers"])

print(eval_dataset)
# ==== EVAL LOOP

model.eval()

torch.set_grad_enabled(False)



# store information for evaluation

future_coords_offsets_pd = []

timestamps = []



agent_ids = []

progress_bar = tqdm(eval_dataloader)

for data in progress_bar:

    _, ouputs = forward(data, model, device, criterion)

    future_coords_offsets_pd.append(ouputs.cpu().numpy().copy())

    timestamps.append(data["timestamp"].numpy().copy())

    agent_ids.append(data["track_id"].numpy().copy())
pred_path = f"{gettempdir()}/pred.csv"



write_pred_csv(pred_path,

               timestamps=np.concatenate(timestamps),

               track_ids=np.concatenate(agent_ids),

               coords=np.concatenate(future_coords_offsets_pd),

              )
metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood, time_displace])

for metric_name, metric_mean in metrics.items():

    print(metric_name, metric_mean)
model.eval()

torch.set_grad_enabled(False)



# build a dict to retrieve future trajectories from GT

gt_rows = {}

for row in read_gt_csv(eval_gt_path):

    gt_rows[row["track_id"] + row["timestamp"]] = row["coord"]



eval_ego_dataset = EgoDataset(cfg, eval_dataset.dataset, rasterizer)



for frame_number in range(99, len(eval_zarr.frames), 100):  # start from last frame of scene_0 and increase by 100

    agent_indices = eval_dataset.get_frame_indices(frame_number) 

    if not len(agent_indices):

        continue



    # get AV point-of-view frame

    data_ego = eval_ego_dataset[frame_number]

    im_ego = rasterizer.to_rgb(data_ego["image"].transpose(1, 2, 0))

    center = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    

    predicted_positions = []

    target_positions = []



    for v_index in agent_indices:

        data_agent = eval_dataset[v_index]



        out_net = model(torch.from_numpy(data_agent["image"]).unsqueeze(0).to(device))

        out_pos = out_net[0].reshape(-1, 2).detach().cpu().numpy()

        # store absolute world coordinates

        predicted_positions.append(out_pos + data_agent["centroid"][:2])

        # retrieve target positions from the GT and store as absolute coordinates

        track_id, timestamp = data_agent["track_id"], data_agent["timestamp"]

        target_positions.append(gt_rows[str(track_id) + str(timestamp)] + data_agent["centroid"][:2])





    # convert coordinates to AV point-of-view so we can draw them

    predicted_positions = transform_points(np.concatenate(predicted_positions), data_ego["world_to_image"])

    target_positions = transform_points(np.concatenate(target_positions), data_ego["world_to_image"])



    yaws = np.zeros((len(predicted_positions), 1))

    draw_trajectory(im_ego, predicted_positions, yaws, PREDICTED_POINTS_COLOR)

    draw_trajectory(im_ego, target_positions, yaws, TARGET_POINTS_COLOR)



    plt.imshow(im_ego[::-1])

    plt.show()