!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py > /dev/null

!python pytorch-xla-env-setup.py --version 20200420 --apt-packages libomp5 libopenblas-dev > /dev/null
import pandas as pd

import torch

from torch.utils.data import Dataset,DataLoader

from tqdm import tqdm

from datetime import datetime

import os

import warnings

from glob import glob

import time



warnings.filterwarnings("ignore")



import torch_xla

import torch_xla.core.xla_model as xm

import torch_xla.distributed.parallel_loader as pl

import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data.distributed import DistributedSampler

from typing import Iterator, List, Optional

from torch.utils.data.sampler import Sampler

from torch.utils.data.dataset import Dataset

from operator import itemgetter

import numpy as np



##################################

## parts of code from catalyst: ##

##################################



class DatasetFromSampler(Dataset):

    """Dataset of indexes from `Sampler`."""



    def __init__(self, sampler: Sampler):

        """

        Args:

            sampler (Sampler): @TODO: Docs. Contribution is welcome

        """

        self.sampler = sampler

        self.sampler_list = None



    def __getitem__(self, index: int):

        """Gets element of the dataset.

        Args:

            index (int): index of the element in the dataset

        Returns:

            Single element by index

        """

        if self.sampler_list is None:

            self.sampler_list = list(self.sampler)

        return self.sampler_list[index]



    def __len__(self) -> int:

        """

        Returns:

            int: length of the dataset

        """

        return len(self.sampler)





class DistributedSamplerWrapper(DistributedSampler):

    """

    Wrapper over `Sampler` for distributed training.

    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with

    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each

    process can pass a DistributedSamplerWrapper instance as a DataLoader

    sampler, and load a subset of subsampled data of the original dataset

    that is exclusive to it.

    .. note::

        Sampler is assumed to be of constant size.

    """



    def __init__(

        self,

        sampler,

        num_replicas: Optional[int] = None,

        rank: Optional[int] = None,

        shuffle: bool = True,

    ):

        """

        Args:

            sampler: Sampler used for subsampling

            num_replicas (int, optional): Number of processes participating in

                distributed training

            rank (int, optional): Rank of the current process

                within ``num_replicas``

            shuffle (bool, optional): If true (default),

                sampler will shuffle the indices

        """

        super(DistributedSamplerWrapper, self).__init__(

            DatasetFromSampler(sampler),

            num_replicas=num_replicas,

            rank=rank,

            shuffle=shuffle,

        )

        self.sampler = sampler



    def __iter__(self):

        """@TODO: Docs. Contribution is welcome."""

        self.dataset = DatasetFromSampler(self.sampler)

        indexes_of_indexes = super().__iter__()

        subsampler_indexes = self.dataset

        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))



class BalanceClassSampler(Sampler):

    """Abstraction over data sampler.

    Allows you to create stratified sample on unbalanced classes.

    """



    def __init__(self, labels: List[int], mode: str = "downsampling"):

        """

        Args:

            labels (List[int]): list of class label

                for each elem in the datasety

            mode (str): Strategy to balance classes.

                Must be one of [downsampling, upsampling]

        """

        super().__init__(labels)



        labels = np.array(labels)

        samples_per_class = {

            label: (labels == label).sum() for label in set(labels)

        }



        self.lbl2idx = {

            label: np.arange(len(labels))[labels == label].tolist()

            for label in set(labels)

        }



        if isinstance(mode, str):

            assert mode in ["downsampling", "upsampling"]



        if isinstance(mode, int) or mode == "upsampling":

            samples_per_class = (

                mode

                if isinstance(mode, int)

                else max(samples_per_class.values())

            )

        else:

            samples_per_class = min(samples_per_class.values())



        self.labels = labels

        self.samples_per_class = samples_per_class

        self.length = self.samples_per_class * len(set(labels))



    def __iter__(self) -> Iterator[int]:

        """

        Yields:

            indices of stratified sample

        """

        indices = []

        for key in sorted(self.lbl2idx):

            replace_ = self.samples_per_class > len(self.lbl2idx[key])

            indices += np.random.choice(

                self.lbl2idx[key], self.samples_per_class, replace=replace_

            ).tolist()

        assert len(indices) == self.length

        np.random.shuffle(indices)



        return iter(indices)



    def __len__(self) -> int:

        """

        Returns:

             length of result sample

        """

        return self.length
class DatasetRetriever(Dataset):



    def __init__(self, df):

        self.labels = df['toxic'].values



    def __len__(self):

        return self.labels.shape[0]



    def __getitem__(self, idx):

        label = self.labels[idx]

        return label

    

    def get_labels(self):

        return list(self.labels )
%%time



df_train = pd.read_csv(f'../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv', index_col='id')
train_dataset = DatasetRetriever(df_train)
train_loader = torch.utils.data.DataLoader(

    train_dataset,

    batch_size=16,

    pin_memory=False,

    drop_last=False,

    num_workers=2

)
result = {'toxic': []}

for labels in tqdm(train_loader, total=len(train_loader)):

    result['toxic'].extend(labels.numpy())
pd.DataFrame(result)['toxic'].hist()
train_loader = torch.utils.data.DataLoader(

    train_dataset,

    batch_size=16,

    sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode='downsampling'),  # here 2 modes: downsampling/upsampling

    pin_memory=False,

    drop_last=False,

    num_workers=2

)
result = {'toxic': []}

for labels in tqdm(train_loader, total=len(train_loader)):

    result['toxic'].extend(labels.numpy())
pd.DataFrame(result)['toxic'].hist()
def run_experiment1(device):

    if not os.path.exists('experiment1'):

        os.makedirs('experiment1')

        

    train_sampler = torch.utils.data.distributed.DistributedSampler(

        train_dataset,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        shuffle=True

    )

    train_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=16,

        sampler=train_sampler,

        pin_memory=False,

        drop_last=False,

        num_workers=1

    )

    para_loader = pl.ParallelLoader(train_loader, [device])



    result = {'toxic': []}

    for labels in para_loader.per_device_loader(device):

        result['toxic'].extend(labels.cpu().numpy())

    pd.DataFrame(result).to_csv(f'experiment1/result_{datetime.utcnow().microsecond}.csv')
def run_experiment2(device):

    if not os.path.exists('experiment2'):

        os.makedirs('experiment2')



    train_sampler = DistributedSamplerWrapper(

        sampler=BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling"),

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        shuffle=True

    )

    train_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=16,

        sampler=train_sampler,

        pin_memory=False,

        drop_last=False,

        num_workers=1

    )

    para_loader = pl.ParallelLoader(train_loader, [device])



    result = {'toxic': []}

    for labels in para_loader.per_device_loader(device):

        result['toxic'].extend(labels.cpu().numpy())

    pd.DataFrame(result).to_csv(f'experiment2/result_{datetime.utcnow().microsecond}.csv')
def _mp_fn(rank, flags):

    device = xm.xla_device()

    run_experiment1(device)

    run_experiment2(device)



FLAGS={}

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
submission = pd.concat([pd.read_csv(path) for path in glob('experiment1/*.csv')])

submission['toxic'].hist()
submission = pd.concat([pd.read_csv(path) for path in glob('experiment2/*.csv')])

submission['toxic'].hist()