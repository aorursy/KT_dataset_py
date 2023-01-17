!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
!cp -rfp ../input/csc-src/src/* .
!pip install pytorch-lightning ase
import dataclasses
from collections import OrderedDict
from os import cpu_count
from pprint import pformat
from time import time
from typing import List, Dict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold

from csc import loader
from csc.const import INPUT_DATA_DIR, EXP_DIR
from csc.dimenet import DimeNet
from csc.loader import AtomsBatch
from mylib.data import PandasDataset
from mylib.dl.opts.ranger import Ranger

# noinspection PyUnresolvedReferences
torch.backends.cudnn.deterministic = True
# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = True


@dataclasses.dataclass
class Conf:
    db_path: str

    gpus: int = 1

    lr: float = 1e-4
    weight_decay: float = 1e-4

    use_16bit: bool = False

    batch_size: int = 64
    epochs: int = 400

    fold: int = 0
    n_splits: int = 4

    seed: int = 0

    def __post_init__(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def __str__(self):
        return pformat(dataclasses.asdict(self))


class Net(pl.LightningModule):
    def __init__(self, hparams: Conf):
        super().__init__()
        self.hparams = hparams
        self.model = DimeNet(
            128,
            num_blocks=6,
            num_bilinear=8,
            num_spherical=7,
            num_radial=6,
            num_targets=4,
        )

    def forward(self, inputs):
        out = self.model(inputs)
        return out

    def training_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, is_train=True)

    def validation_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, is_train=False)

    def training_epoch_end(self, outputs):
        return self.__epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        return self.__epoch_end(outputs, 'val')

    # noinspection PyUnusedLocal
    def __step(self, batch, batch_idx, is_train: bool):
        batch = AtomsBatch(**batch)
        out = self.forward(batch)
        loss = lmae_loss(out.y_pred, out.y_true)
        mae = torch.abs(out.y_true - out.y_pred).mean()

        prefix = 'train' if is_train else 'val'
        out = {
            'loss': loss,
            f'{prefix}_loss': loss,
            f'{prefix}_mae': mae,
            f'{prefix}_size': out.n_pairs,
        }
        if is_train:
            out = {'loss': loss, **out}

        return OrderedDict(out)

    def __epoch_end(self, outputs: List[Dict], prefix: str):
        loss = 0
        mae = 0
        total_size = 0

        for o in outputs:
            loss += o[f'{prefix}_loss'] * o[f'{prefix}_size']
            mae += o[f'{prefix}_mae'] * o[f'{prefix}_size']
            total_size += o[f'{prefix}_size']

        loss = loss / total_size
        # noinspection PyTypeChecker
        lmae = torch.log(mae / total_size)

        # Skip sanity check
        if not (prefix == 'val' and self.current_epoch == 0):
            self.logger.experiment.add_scalars(f'loss', {
                prefix: loss,
            }, self.current_epoch)
            self.logger.experiment.add_scalars(f'lmae', {
                prefix: lmae,
            }, self.current_epoch)

        return OrderedDict({
            'progress_bar': {
                f'{prefix}_loss': loss,
            },
        })

    def configure_optimizers(self):
        opt = Ranger(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return opt

    # noinspection PyAttributeOutsideInit
    def prepare_data(self):
        df = pd.read_pickle(self.hparams.db_path)

        folds = KFold(n_splits=self.hparams.n_splits, random_state=self.hparams.seed, shuffle=True)
        train_idx, val_idx = list(folds.split(df))[self.hparams.fold]

        self.df_train = df.iloc[train_idx]
        self.df_val = df.iloc[val_idx]

    def train_dataloader(self):
        return loader.get_loader(
            PandasDataset(self.df_train),
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=True,
            cutoff=5.,
        )

    def val_dataloader(self):
        return loader.get_loader(
            PandasDataset(self.df_val),
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=True,
            cutoff=5.,
        )


def lmae_loss(y_pred, y_true):
    loss = torch.abs(y_true - y_pred)
    loss = loss.mean(dim=0)
    loss = torch.log(loss)

    return loss


class MyTensorBoardLogger(TensorBoardLogger):
    def _convert_params(self, params: Conf):
        return dataclasses.asdict(params)


def main(conf: Conf):
    model = Net(conf)

    logger = MyTensorBoardLogger(
        EXP_DIR,
        name='mol',
        version=str(time()),
    )

    trainer = pl.Trainer(
        max_epochs=conf.epochs,
#         gpus=conf.gpus,
        num_tpu_cores=1,
        logger=logger,
        precision=16 if conf.use_16bit else 32,
    )
    trainer.fit(model)


if __name__ == '__main__':
    main(Conf(
        db_path=str(INPUT_DATA_DIR / '1JHN.pkl'),

        lr=1e-4,
        batch_size=32,
    ))

