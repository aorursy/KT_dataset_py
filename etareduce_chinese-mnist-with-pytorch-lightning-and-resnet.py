!pip install -U -q --use-feature=2020-resolver "pytorch_lightning==0.10.0rc1"
import os

from pathlib import Path



import albumentations as A

import cv2

import numpy as np

import pandas as pd

import pytorch_lightning as pl

import torch

from albumentations.pytorch import ToTensorV2

from albumentations.augmentations.transforms import Blur, RandomBrightness

from torch.utils.data import DataLoader, Dataset





class ChineseMNISTDataset(Dataset):

    def __init__(

        self,

        df: pd.DataFrame,

        image_root: Path,

        transform: A.BasicTransform = None,

    ) -> None:

        super().__init__()

        self.df = df

        self.image_root = image_root

        self.transform = transform



    def __getitem__(self, idx: int):

        row = self.df.loc[idx, :]

        suite_id, code, sample_id = row.suite_id, row.code, row.sample_id

        filename = self.image_root / f"input_{suite_id}_{sample_id}_{code}.jpg"

        assert os.path.isfile(filename), f"{filename} is not a file"

        image = cv2.imread(str(filename))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = image[:, np.newaxis]

        if self.transform is not None:

            image = self.transform(image=image)["image"]

        return image, code - 1



    def __len__(self):

        return len(self.df)





class ChineseMNISTDataModule(pl.LightningDataModule):

    def __init__(

        self,

        data_root: Path,

        all_df: pd.DataFrame,

        train_indices: pd.Index,

        val_indices: pd.Index,

    ) -> None:

        super().__init__()

        self.data_root = data_root

        self.df = all_df

        self.image_root = self.data_root / "data" / "data"

        self.train_df = self.df.loc[train_indices, :].copy().reset_index()

        self.train_transform = A.Compose(

            [

                Blur(),

                RandomBrightness(),

                ToTensorV2(),

            ]

        )

        self.val_df = self.df.loc[val_indices, :].copy().reset_index()

        self.val_transform = A.Compose(

            [

                ToTensorV2(),

            ]

        )



    def train_dataloader(self):

        ds = ChineseMNISTDataset(self.train_df, self.image_root, self.train_transform)

        return DataLoader(

            ds,

            batch_size=64,

            shuffle=True,

            num_workers=4,

            pin_memory=True,

        )



    def val_dataloader(self):

        ds = ChineseMNISTDataset(self.val_df, self.image_root, self.val_transform)

        return DataLoader(

            ds,

            batch_size=64,

            shuffle=False,

            num_workers=4,

            pin_memory=True,

        )





if __name__ == "__main__":

    is_kaggle = os.path.isdir("/kaggle")

    data_root = Path("/kaggle/input/chinese-mnist" if is_kaggle else "archive")

    assert os.path.isdir(data_root), f"{data_root} is not a dir"

    df = pd.read_csv(data_root / "chinese_mnist.csv")



    data_module = ChineseMNISTDataModule(data_root, df, df.index[:20], df.index[20:30])

import os

from pathlib import Path



import pandas as pd

import pytorch_lightning as pl

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from pytorch_lightning.metrics import Accuracy

from sklearn.model_selection import StratifiedKFold

from torch import nn, optim

from torchvision.models import resnet18



try:

    from dataset import ChineseMNISTDataModule, ChineseMNISTDataset

except:

    pass





class ChineseMNISTResnetModel(pl.LightningModule):

    def __init__(self, learning_rate=1e-3):

        super().__init__()

        self.learning_rate = learning_rate

        self.num_classes = 15

        resnet = resnet18(pretrained=True, progress=True)

        resnet.conv1 = nn.Conv2d(

            in_channels=1,

            out_channels=resnet.conv1.out_channels,

            kernel_size=resnet.conv1.kernel_size,

            stride=resnet.conv1.stride,

            dilation=resnet.conv1.dilation,

            bias=resnet.conv1.bias,

        )

        resnet.fc = nn.Linear(512, self.num_classes)

        self.resnet = resnet

        self.accuracy = Accuracy(num_classes=self.num_classes)

        self.criterion = nn.CrossEntropyLoss()



    def forward(self, image):

        image = image.permute(0, 3, 1, 2).contiguous().float()

        return self.resnet(image)



    def training_step(self, batch, batch_idx: int):

        image, y = batch

        yhat = self(image)

        loss = self.criterion(yhat, y)

        acc = self.accuracy(yhat, y)

        return {"loss": loss, "acc": acc}



    def validation_step(self, batch, batch_idx: int):

        image, y = batch

        yhat = self(image)

        loss = self.criterion(yhat, y)

        acc = self.accuracy(yhat, y)

        return {"val_loss": loss, "val_acc": acc, "progress_bar": {"val_acc": acc}}



    def test_step(self, batch, batch_idx):

        metrics = self.validation_step(batch, batch_idx)

        return {"test_acc": metrics["val_acc"], "test_loss": metrics["val_loss"]}



    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer





def training(k_folds: int = 5):

    is_kaggle = os.path.isdir("/kaggle")

    data_root = Path("/kaggle/input/chinese-mnist" if is_kaggle else "archive")

    all_df = pd.read_csv(data_root / "chinese_mnist.csv")



    skf = StratifiedKFold(n_splits=k_folds, shuffle=True)



    checkpoint_callback = ModelCheckpoint(

        filepath=os.getcwd(),

        save_top_k=1,

        verbose=True,

        monitor="val_loss",

        mode="min",

    )

    trainer = pl.Trainer(

        gpus=1,

        max_epochs=4,

        precision=16,

        val_check_interval=0.2,

        checkpoint_callback=checkpoint_callback,

    )



    for train_indices, val_indices in skf.split(all_df, all_df.code):

        data_module = ChineseMNISTDataModule(

            data_root=data_root,

            all_df=all_df,

            train_indices=train_indices,

            val_indices=val_indices,

        )

        model = ChineseMNISTResnetModel()

        trainer.fit(model, data_module)





if __name__ == "__main__":

    training()
