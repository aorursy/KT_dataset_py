!pip install -q -U pip==20.2.3
!pip uninstall -y typing

!pip install -q l5kit==1.1 pytorch-lightning==0.10.0
import l5kit

import torch

import torchvision

import pytorch_lightning as pl

l5kit.__version__, torch.__version__, torchvision.__version__, pl.__version__, torch.cuda.is_available()
import numpy as np

import torch

from l5kit.geometry import transform_points

from torch import Tensor





def convert_agent_coordinates_to_world_offsets(

    agents_coords: np.ndarray,

    world_from_agents: np.ndarray,

    centroids: np.ndarray,

) -> np.ndarray:

    coords_offset = []

    for agent_coords, world_from_agent, centroid in zip(

        agents_coords, world_from_agents, centroids

    ):

        predition_offset = []

        for agent_coord in agent_coords:

            predition_offset.append(

                transform_points(agent_coord, world_from_agent) - centroid[:2]

            )

        predition_offset = np.stack(predition_offset)

        coords_offset.append(predition_offset)

    return np.stack(coords_offset)





def pytorch_neg_multi_log_likelihood_batch(

    gt: Tensor,

    pred: Tensor,

    confidences: Tensor,

    avails: Tensor,

) -> Tensor:

    """

    Compute a negative log-likelihood for the multi-modal scenario.

    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:

    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    https://leimao.github.io/blog/LogSumExp/

    Args:

        gt (Tensor): array of shape (bs)x(time)x(2D coords)

        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)

        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample

        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep

    Returns:

        Tensor: negative log-likelihood for this example, a single float number

    """

    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"

    batch_size, num_modes, future_len, num_coords = pred.shape



    assert gt.shape == (

        batch_size,

        future_len,

        num_coords,

    ), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"

    assert confidences.shape == (

        batch_size,

        num_modes,

    ), f"expected 1D (Modes) array for confidences, got {confidences.shape}"

    assert torch.allclose(

        torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))

    ), "confidences should sum to 1"

    assert avails.shape == (

        batch_size,

        future_len,

    ), f"expected 1D (Time) array for avails, got {avails.shape}"

    # assert all data are valid

    assert torch.isfinite(pred).all(), "invalid value found in pred"

    assert torch.isfinite(gt).all(), "invalid value found in gt"

    assert torch.isfinite(confidences).all(), "invalid value found in confidences"

    assert torch.isfinite(avails).all(), "invalid value found in avails"



    # convert to (batch_size, num_modes, future_len, num_coords)

    gt = torch.unsqueeze(gt, 1)  # add modes

    avails = avails[:, None, :, None]  # add modes and cords



    # error (batch_size, num_modes, future_len)

    error = torch.sum(

        ((gt - pred) * avails) ** 2, dim=-1

    )  # reduce coords and use availability



    with np.errstate(

        divide="ignore"

    ):  # when confidence is 0 log goes to -inf, but we're fine with it

        # error (batch_size, num_modes)

        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time



    # use max aggregator on modes for numerical stability

    # error (batch_size, num_modes)

    max_value, _ = error.max(

        dim=1, keepdim=True

    )  # error are negative at this point, so max() gives the minimum one

    error = (

        -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True))

        - max_value

    )  # reduce modes

    # print("error", error)

    return torch.mean(error)

import os

from typing import Dict, Optional



import matplotlib.pyplot as plt

import numpy as np

import pytorch_lightning as pl

from l5kit.data import ChunkedDataset, LocalDataManager

from l5kit.dataset import AgentDataset

from l5kit.geometry import transform_points

from l5kit.rasterization import build_rasterizer

from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory

from torch.utils.data import DataLoader



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

        "key": "scenes/train.zarr",

        "batch_size": 24,

        "shuffle": True,

        "num_workers": 4,

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





class LyftAgentDataModule(pl.LightningDataModule):

    def __init__(

        self,

        cfg: Dict = CONFIG_DATA,

        data_root: str = data_root,

    ):

        super().__init__()

        self.cfg = cfg

        self.dm = LocalDataManager(data_root)

        self.rast = build_rasterizer(self.cfg, self.dm)

        # self.ego_dataset = EgoDataset(self.cfg, self.zarr_dataset, self.rast)



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



    def plt_show_agent_map(self, idx):

        data = self.agent_dataset[idx]

        im = data["image"].transpose(1, 2, 0)

        im = self.rast.to_rgb(im)

        target_positions_pixels = transform_points(

            data["target_positions"] + data["centroid"][:2], data["world_to_image"]

        )

        draw_trajectory(

            im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR

        )

        plt.imshow(im[::-1])

        plt.show()

import os

from pathlib import Path

from typing import Dict



import numpy as np

import pytorch_lightning as pl

import torch

from l5kit.evaluation.csv_utils import write_pred_csv

from pytorch_lightning.callbacks import ModelCheckpoint

from torch import nn, optim

from torchvision.models.resnet import resnet18, resnet34, resnet50

from tqdm.notebook import tqdm



try:

    from jiayu_model.data_module import LyftAgentDataModule

    from jiayu_model.loss_function import (

        convert_agent_coordinates_to_world_offsets,

        pytorch_neg_multi_log_likelihood_batch,

    )

except:

    pass





class BaselineModel(pl.LightningModule):

    """Our baseline is a simple resnet pretrained on imagenet.

    We must replace the input and the final layer to address our requirements.

    """



    resnet_builders = {

        "resnet18": resnet18,

        "resnet34": resnet34,

        "resnet50": resnet50,

    }

    backbone_out_feature_counts = {

        "resnet18": 512,

        "resnet34": 512,

        "resnet50": 2048,

    }



    def __init__(

        self,

        cfg: Dict,

        learning_rate=1e-3,

        num_modes: int = 3,

        pretrained=False,

    ):

        super().__init__()

        self.save_hyperparameters()



        cfg = self.hparams.cfg  # type: ignore

        self.learning_rate = self.hparams.learning_rate  # type: ignore

        self.num_modes = self.hparams.num_modes  # type: ignore

        self.criterion = pytorch_neg_multi_log_likelihood_batch



        model_architecture = cfg["model_params"]["model_architecture"]

        resnet_builder = BaselineModel.resnet_builders[model_architecture]

        backbone_out_features = BaselineModel.backbone_out_feature_counts[

            model_architecture

        ]



        resnet = resnet_builder(pretrained=pretrained)

        # change input channels number to match the rasterizer's output

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2

        num_in_channels = 3 + num_history_channels

        resnet.conv1 = nn.Conv2d(

            num_in_channels,

            resnet.conv1.out_channels,

            kernel_size=resnet.conv1.kernel_size,

            stride=resnet.conv1.stride,

            padding=resnet.conv1.padding,

            bias=resnet.conv1.bias,

        )

        self.future_len = cfg["model_params"]["future_num_frames"]

        num_targets = 2 * self.future_len

        self.num_preds = num_targets * self.num_modes

        resnet.fc = nn.Sequential(

            nn.Linear(

                in_features=backbone_out_features,

                # num of modes * preds + confidence

                out_features=self.num_preds + self.num_modes,

            ),

        )

        self.resnet = resnet



    def forward(self, data):

        out = self.resnet(data)

        batch_size = data.shape[0]

        pred, confidences = torch.split(out, self.num_preds, dim=1)

        assert pred.shape == (batch_size, self.num_preds)

        assert confidences.shape == (batch_size, self.num_modes)

        pred = pred.view(batch_size, self.num_modes, self.future_len, 2)

        confidences = torch.softmax(confidences, dim=1)

        return pred, confidences



    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer



    def training_step(self, batch, batch_idx: int):  # type: ignore

        target_availabilities = batch["target_availabilities"]

        targets = batch["target_positions"]

        data = batch["image"]

        pred, confidences = self(data)

        loss = self.criterion(targets, pred, confidences, target_availabilities)

        return loss



    def validation_step(self, batch, batch_idx: int):  # type: ignore

        target_availabilities = batch["target_availabilities"]

        targets = batch["target_positions"]

        data = batch["image"]

        pred, confidences = self(data)

        loss = self.criterion(targets, pred, confidences, target_availabilities)

        self.log("val_loss", loss)





def train_model(

    lyft_agent_data_module: LyftAgentDataModule,

    max_num_steps: int,

    eval_every_n_steps: int,

    max_epochs: int = 2,

    fast_dev_run=False,

) -> BaselineModel:

    model = BaselineModel(cfg=lyft_agent_data_module.cfg)

    checkpoint_callback = ModelCheckpoint(

        filepath=str(Path(os.getcwd()) / "baseline_checkpoints"),

        save_top_k=1,

        verbose=True,

        monitor="val_loss",

        mode="min",

    )

    trainer = pl.Trainer(

        gpus=1,

        # tpu_cores=1,

        precision=16,

        max_epochs=max_epochs,

        max_steps=max_num_steps,

        checkpoint_callback=checkpoint_callback,

        fast_dev_run=fast_dev_run,

        val_check_interval=eval_every_n_steps / max_num_steps,

        limit_train_batches=max_num_steps // max_epochs,

        limit_val_batches=100,

    )

    trainer.fit(model=model, datamodule=lyft_agent_data_module)

    return model





def evaluation(checkpoint_path: str, lyft_agent_data_module: LyftAgentDataModule):

    assert torch.cuda.is_available(), "GPU must be used"

    device = torch.device("cuda")

    model = BaselineModel.load_from_checkpoint(checkpoint_path, map_location=device)

    model = model.to(device)

    model.eval()

    model.freeze()



    pred_coords_list = []

    confidences_list = []

    timestamps_list = []

    track_id_list = []



    for data in tqdm(lyft_agent_data_module.test_dataloader()):

        pred, confidences = model(data["image"].to(device))

        pred = convert_agent_coordinates_to_world_offsets(

            pred.detach().cpu().numpy(),

            data["world_from_agent"].numpy(),

            data["centroid"].numpy(),

        )

        pred_coords_list.append(pred)

        confidences_list.append(confidences.detach().cpu().numpy())

        timestamps_list.append(data["timestamp"].detach().numpy())

        track_id_list.append(data["track_id"].detach().numpy())



    timestamps = np.concatenate(timestamps_list)

    track_ids = np.concatenate(track_id_list)

    coords = np.concatenate(pred_coords_list)

    confs = np.concatenate(confidences_list)



    return timestamps, track_ids, coords, confs





if __name__ == "__main__":

    print("loading dataset")

    lyft_agent_data_module = LyftAgentDataModule()  # type: ignore

    cfg = lyft_agent_data_module.cfg



    mode = "training"



    if mode == "training":

        max_num_steps = cfg["train_params"]["max_num_steps"]

        eval_every_n_steps = cfg["train_params"]["eval_every_n_steps"]

        print(f"max_num_steps={max_num_steps}, eval_every_n_steps={eval_every_n_steps}")



        model_save_path = "model_state_last.pth"

        print("dataset is loaded, starting training")

        model = train_model(

            lyft_agent_data_module,

            max_num_steps=max_num_steps,

            eval_every_n_steps=eval_every_n_steps,

        )

        print("training done, saving state and generating results")

        torch.save(model.state_dict(), model_save_path)

        print(f"saving done, model path is {model_save_path}")

    elif mode == "evaluation":

        checkpoint_path = (

            "../input/lyft-av-py-lightning-resnet-training-baseline/epoch=4.ckpt"

        )

        timestamps, track_ids, coords, confs = evaluation(

            checkpoint_path,

            lyft_agent_data_module,

        )

        write_pred_csv(

            "submission.csv",

            timestamps=timestamps,

            track_ids=track_ids,

            coords=coords,

            confs=confs,

        )
