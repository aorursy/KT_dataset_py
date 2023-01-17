!pip install -q efficientnet_pytorch > /dev/null
import os

import gc

import copy

import random

import time

import traceback

import numpy as np

import pandas as pd

from tqdm import tqdm

from glob import glob

from pathlib import Path

from datetime import datetime

from sklearn.metrics import roc_auc_score

from typing import Callable, List, Dict, Tuple, Union, Any



import cv2

import torch

import torch.nn as nn

from PIL import Image

from torch import Tensor

import albumentations as A

from dataclasses import dataclass

from torchvision import transforms

from efficientnet_pytorch import EfficientNet

from torch.utils.data import Dataset, DataLoader

from albumentations.pytorch.transforms import ToTensorV2



import warnings

warnings.filterwarnings("ignore")
@dataclass

class Project:

    """

    This class stores information about the paths.

    """



    kaggle_path: Path = Path(".").absolute().parent

    data_dir = kaggle_path / "input/alaska2-image-steganalysis"

    output_dir = kaggle_path / "working"

    checkpoint_dir = output_dir / "checkpoint"



    def __post_init__(self):

        # create the directories if they don't exist

        self.checkpoint_dir.mkdir(exist_ok=True)





alaska = Project()
def train_val_split() -> pd.DataFrame:



    alg_mapping = {"JMiPOD": 0, "JUNIWARD": 1, "UERD": 2, "Cover": 3}



    train_files = os.listdir(alaska.data_dir / "Cover")



    df_pos_train_split = pd.DataFrame(train_files)

    df_pos_train_split.rename(columns={0: "image"}, inplace=True)

    df_pos_train_split["alg_idx"] = df_pos_train_split.index // 25_000

    df_pos_train_split["folds"] = np.concatenate([(np.arange(25_000) // 5000)] * 3)



    JMiPOD_files = df_pos_train_split[

        df_pos_train_split.alg_idx == alg_mapping["JMiPOD"]

    ].image.values

    JUNIWARD_files = df_pos_train_split[

        df_pos_train_split.alg_idx == alg_mapping["JUNIWARD"]

    ].image.values

    UERD_files = df_pos_train_split[

        df_pos_train_split.alg_idx == alg_mapping["UERD"]

    ].image.values



    path1 = [(alaska.data_dir / "JMiPOD" / i).as_posix() for i in JMiPOD_files]

    path2 = [(alaska.data_dir / "JUNIWARD" / i).as_posix() for i in JUNIWARD_files]

    path3 = [(alaska.data_dir / "UERD" / i).as_posix() for i in UERD_files]



    df_pos_train_split["img_path"] = np.concatenate(np.array([path1, path2, path3]))



    df_pos = df_pos_train_split.assign(

        img_path=np.concatenate(np.array([path1, path2, path3])), label=1

    )



    df_neg = df_pos[["image", "folds"]].copy()

    df_neg = df_neg.assign(

        img_path=[(alaska.data_dir / "Cover" / i).as_posix() for i in df_neg.image],

        label=0,

    )



    cols = ["image", "folds", "img_path", "label"]

    df_train = pd.concat([df_pos[cols], df_neg[cols]])

    df_train = df_train.sample(frac=1.0, random_state=101).reset_index(drop=True)

    del df_pos_train_split, df_pos, df_neg

    _ = gc.collect()



    return df_train
def get_train_transforms() -> Callable:

    """

    source: https://www.kaggle.com/shonenkov/train-inference-gpu-baseline

    """

    return A.Compose(

        [

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.5),

            A.Resize(height=512, width=512, p=1.0),

            ToTensorV2(),

        ],

        p=1.0,

    )



def get_valid_transforms() -> Callable:

    """

    source: https://www.kaggle.com/shonenkov/train-inference-gpu-baseline

    """

    return A.Compose([A.Resize(height=512, width=512, p=1.0), ToTensorV2(),], p=1.0)





transform_strategy = {"train": get_train_transforms(), "val": get_valid_transforms()}
class AlaskaClassifierDataset(Dataset):

    """

    source: https://www.kaggle.com/shonenkov/train-inference-gpu-baseline

    """



    def __init__(

        self, data: pd.DataFrame, transform: dict, val_fold: int = 0, train: bool = True

    ):



        super().__init__()

        self.train = train

        self.val_fold = val_fold

        if self.train:

            self.data = data[data["folds"] != self.val_fold].reset_index(drop=True)

        else:

            self.data = data[data["folds"] == self.val_fold].reset_index(drop=True)



        self.transform = transform["train"] if self.train else transform["val"]



    def __len__(self) -> int:

        return len(self.data)



    def __getitem__(self, idx: int) -> Dict[torch.Tensor, torch.Tensor]:

        image = cv2.imread(self.data.loc[idx, "img_path"], cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0

        if self.transform:

            sample = {"image": image}

            sample = self.transform(**sample)

            image = sample["image"]



        target = self.data.loc[idx, "label"]



        return {"image": image, "target": target}
df_train = train_val_split()

train_dataset = AlaskaClassifierDataset(df_train, transform_strategy)

val_dataset = AlaskaClassifierDataset(df_train, transform_strategy, train=False)
def build_model(pretrained_model: str = "efficientnet-b3") -> nn.Module:

    model = EfficientNet.from_name(pretrained_model)

    n_input_feats = model._fc.in_features

    model._fc = nn.Linear(n_input_feats, 1)

    return model





model = build_model()
!git clone https://github.com/openai/spinningup.git -q

%cd spinningup

!pip install -q -e . > /dev/null
try: 

    from spinup.utils.logx import EpochLogger

except:

    print('ones more..')

    from spinup.utils.logx import EpochLogger
def to_numpy(tensor: Union[Tensor, Image.Image, np.array]) -> np.ndarray:

    """

    source: https://www.kaggle.com/sermakarevich/complete-handcrafted-pipeline-in-pytorch-resnet9

    """

    if type(tensor) == np.array or type(tensor) == np.ndarray:

        return np.array(tensor)

    elif type(tensor) == Image.Image:

        return np.array(tensor)

    elif type(tensor) == Tensor:

        return tensor.cpu().detach().numpy()

    else:

        raise ValueError(msg)





def copy_data_to_device(data: Any, device: str) -> None:

    if torch.is_tensor(data):

        return data.to(device, dtype=torch.float)

    elif isinstance(data, (list, tuple)):

        return [copy_data_to_device(elem, device) for elem in data]

    raise ValueError("Invalid data type {}".format(type(data)))





def get_auc(y_true: np.array, y_hat: np.array) -> float:

    try:

        auc = roc_auc_score(np.vstack(y_true), np.vstack(y_hat))

    except:

        auc = -1

    return auc
LOSS_FN = nn.BCEWithLogitsLoss()

LR_SCHEDULER = lambda optim: torch.optim.lr_scheduler.ReduceLROnPlateau(

    optim, patience=3, factor=0.5, verbose=True

)
def train_eval_loop(

    model=model,

    train_dataset=train_dataset,

    val_dataset=val_dataset,

    criterion=LOSS_FN,

    lr=1e-4,

    epoch_n=3,

    batch_size=8,

    device=None,

    early_stopping_patience=10,

    l2_reg_alpha=0.0,

    max_batches_per_epoch_train=500,

    max_batches_per_epoch_val=500,

    data_loader_ctor=DataLoader,

    optimizer_ctor=None,

    lr_scheduler_ctor=LR_SCHEDULER,

    shuffle_train=False,

    dataloader_workers_n=1,

):



    logger_kwargs = {

        "output_dir": (alaska.checkpoint_dir).as_posix(),

        "output_fname": "alaska_progress.txt",

        "exp_name": "val_fold_0",

    }



    logger = EpochLogger(**logger_kwargs)

    logger.save_config(locals())

    start_time = time.time()



    if device is None:

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("device: ", device)

    device = torch.device(device)



    model.to(device)



    if optimizer_ctor is None:

        optimizer = torch.optim.Adam(

            model.parameters(), lr=lr, weight_decay=l2_reg_alpha

        )

    else:

        optimizer = optimizer_ctor(model.parameters(), lr=lr)



    if lr_scheduler_ctor is not None:

        lr_scheduler = lr_scheduler_ctor(optimizer)

    else:

        lr_scheduler = None



    train_dataloader = data_loader_ctor(

        train_dataset,

        batch_size=batch_size,

        shuffle=shuffle_train,

        num_workers=dataloader_workers_n,

    )

    val_dataloader = data_loader_ctor(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=dataloader_workers_n,

    )



    best_val_loss = float("inf")

    auc_valid = 0

    best_epoch_i = 0

    best_model = copy.deepcopy(model)



    for epoch_i in range(epoch_n):

        try:

            epoch_start = datetime.now()

            print("epoch {}".format(epoch_i + 1))



            # *************** training part ***********************

            mean_train_loss = 0

            train_batches_n = 0

            y_true_train, y_pred_train = [], []

            model.train()

            pbar = tqdm(

                enumerate(train_dataloader),

                total=max_batches_per_epoch_train,

                desc="Epoch {}".format(epoch_i),

                ncols=0,

            )



            for batch_i, d in pbar:

                if batch_i > max_batches_per_epoch_train:

                    break



                batch_x = copy_data_to_device(d["image"], device)

                batch_y = copy_data_to_device(d["target"].view(-1, 1), device)

                pred = model(batch_x)



                y_true_train.append(to_numpy(batch_y))

                y_pred_train.append(to_numpy(pred))



                loss = criterion(pred, batch_y)



                model.zero_grad()

                loss.backward()



                optimizer.step()



                mean_train_loss += float(loss)

                train_batches_n += 1



            mean_train_loss /= train_batches_n

            score = get_auc(y_true_train, y_pred_train)

            logger.store(TrainLoss=mean_train_loss, TrainAUC=score)



            # logging training info

            Mode_train = "*******"

            logger.log_tabular("Mode_train", Mode_train)

            logger.log_tabular("Epoch", epoch_i + 1)

            logger.log_tabular("TrainLoss", average_only=True)

            logger.log_tabular("TrainAUC", average_only=True)

            logger.log_tabular(

                "TotalGradientSteps", (epoch_i + 1) * max_batches_per_epoch_train

            )



            # *************** eval part ***********************

            model.eval()

            mean_val_loss = 0

            val_batches_n = 0



            total_samples, correct = 0, 0

            y_true_valid, y_pred_valid = [], []



            with torch.no_grad():

                for batch_i, d in enumerate(val_dataloader):

                    if batch_i > max_batches_per_epoch_val:

                        break



                    batch_x = copy_data_to_device(d["image"], device)

                    batch_y = copy_data_to_device(d["target"].view(-1, 1), device)

                    pred = model(batch_x)



                    y_true_valid.append(to_numpy(batch_y))

                    y_pred_valid.append(to_numpy(pred))



                    loss = criterion(pred, batch_y)

                    mean_val_loss += float(loss)



                    val_batches_n += 1



                    correct += pred.eq(batch_y.view_as(pred)).sum().item()

                    total_samples += batch_x.size()[0]



                score = get_auc(y_true_valid, y_pred_valid)

                accuracy = 100.0 * correct / total_samples

                mean_val_loss /= val_batches_n

                logger.store(Loss=mean_val_loss, AUC=score, Accuracy=accuracy)



            if mean_val_loss < best_val_loss:

                best_epoch_i = epoch_i

                best_val_loss = mean_val_loss

                best_model = copy.deepcopy(model)

                logger.setup_pytorch_saver(best_model)  # Setup model saving

                print("New best model!")

            elif epoch_i - best_epoch_i > early_stopping_patience:

                print(

                    "The model has not improved over the past {} epochs, stop training".format(

                        early_stopping_patience

                    )

                )

                break



            if lr_scheduler is not None:

                lr_scheduler.step(mean_val_loss)



            # logging valid info

            Mode_valid = "*******"

            logger.log_tabular("Mode_valid", Mode_valid)

            logger.log_tabular("Loss", average_only=True)

            logger.log_tabular("AUC", average_only=True)

            logger.log_tabular("Accuracy", average_only=True)

            logger.log_tabular(

                "Total time",

                datetime.utcfromtimestamp(time.time() - start_time).strftime("%M:%S"),

            )

            logger.dump_tabular()



            # exception handling

            print()

        except KeyboardInterrupt:

            print("Stopped by user")

            break

        except Exception as ex:

            print("Training error: {}\n{}".format(ex, traceback.format_exc()))

            break
train_eval_loop()