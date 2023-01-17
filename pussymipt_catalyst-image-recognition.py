!pip install catalyst --upgrade
from pathlib import Path

import pandas as pd

import numpy as np

from random import choices, sample

from collections import Counter



from plotly import graph_objects as go



from sklearn.model_selection import train_test_split



import torch

from torch import nn

from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import Sampler

import albumentations as A



from catalyst import dl

from catalyst import utils

from catalyst import data

from catalyst.contrib.nn.criterion import TripletMarginLossWithSampler



import cv2
utils.set_global_seed(42)
def find_value_ids(it, value):

    """

    Args:

        it: list of any

        value: query element



    Returns:

        indices of the all elements equal x0

    """

    if isinstance(it, np.ndarray):

        inds = list(np.where(it == value)[0])

    else:  # could be very slow

        inds = [i for i, el in enumerate(it) if el == value]

    return inds





class BalanceBatchSampler(Sampler):

    """

    This kind of sampler can be used for both metric learning and

    classification task.

    Sampler with the given strategy for the C unique classes dataset:

    - Selection P of C classes for the 1st batch

    - Selection K instances for each class for the 1st batch

    - Selection P of C - P remaining classes for 2nd batch

    - Selection K instances for each class for the 2nd batch

    - ...

    The epoch ends when there are no classes left.

    So, the batch sise is P * K except the last one.

    Thus, in each epoch, all the classes will be selected once, but this

    does not mean that all the instances will be selected during the epoch.

    One of the purposes of this sampler is to be used for

    forming triplets and pos/neg pairs inside the batch.

    To guarante existance of these pairs in the batch,

    P and K should be > 1. (1)

    Behavior in corner cases:

    - If a class does not contain K instances,

    a choice will be made with repetition.

    - If C % P == 1 then one of the classes should be dropped

    otherwise statement (1) will not be met.

    This type of sampling can be found in the classical paper of Person Re-Id,

    where P equals 32 and K equals 4:

    `In Defense of the Triplet Loss for Person Re-Identification`_.

    .. _In Defense of the Triplet Loss for Person Re-Identification:

        https://arxiv.org/abs/1703.07737

    """



    def __init__(self, labels, p: int, k: int):

        """

        Args:

            labels: list of classes labeles for each elem in the dataset

            p: number of classes in a batch, should be > 1

            k: number of instances of each class in a batch, should be > 1

        """

        super().__init__(self)

        classes = set(labels)



        assert isinstance(p, int) and isinstance(k, int)

        assert (1 < p <= len(classes)) and (1 < k)

        assert all(

            n > 1 for n in Counter(labels).values()

        ), "Each class shoud contain at least 2 instances to fit (1)"



        self._labels = labels

        self._p = p

        self._k = k



        self._batch_size = self._p * self._k

        self._classes = classes



        # to satisfy statement (1)

        num_classes = len(self._classes)

        if num_classes % self._p == 1:

            self._num_epoch_classes = num_classes - 1

        else:

            self._num_epoch_classes = num_classes



    @property

    def batch_size(self) -> int:

        """

        Returns:

            this value should be used in DataLoader as batch size

        """

        return self._batch_size



    @property

    def batches_in_epoch(self) -> int:

        """

        Returns:

            number of batches in an epoch

        """

        return int(np.ceil(self._num_epoch_classes / self._p))



    def __len__(self) -> int:

        """

        Returns:

            number of samples in an epoch

        """

        return self._num_epoch_classes * self._k



    def __iter__(self):

        """

        Returns:

            indeces for sampling dataset elems during an epoch

        """

        inds = []



        for cls_id in sample(self._classes, self._num_epoch_classes):

            all_cls_inds = find_value_ids(self._labels, cls_id)



            # we've checked in __init__ that this value must be > 1

            num_samples_exists = len(all_cls_inds)



            if num_samples_exists < self._k:

                selected_inds = sample(

                    all_cls_inds, k=num_samples_exists

                ) + choices(all_cls_inds, k=self._k - num_samples_exists)

            else:

                selected_inds = sample(all_cls_inds, k=self._k)



            inds.extend(selected_inds)



        return iter(inds)

train_df = pd.read_csv("../input/landmark-recognition-2020/train.csv")

train_df.head()
def load_img(id_: str, train=True):

    if train:

        root = Path("../input/landmark-recognition-2020/train")

    else:

        root = Path("../input/landmark-recognition-2020/test")

    first_folder = root / str(id_[0])

    second_folder = first_folder / str(id_[1])

    third_folder = second_folder / str(id_[2])

    path_to_img = third_folder / str(id_)

    img = cv2.imread(str(path_to_img)+".jpg")

    return img
class ImgDataset(Dataset):

    def __init__(self, df, transforms = None, train: bool = True):

        self.id = df.id.values

        if train:

            self.labels = df.landmark_id.values

        self.train = train

        if transforms is None:

            transforms = A.Compose([

                A.Resize(width=224, height=224), 

                A.pytorch.ToTensor()

            ])

        self.transforms = transforms

        

    def __getitem__(self, idx: int):

        img = load_img(self.id[idx], train=self.train)

        tensor_img = self.transforms(image=img)["image"]

        

        output = {"features": tensor_img}

        if self.train:

            label = self.labels[idx]

            output["targets"] = label

        return output

    

    def __len__(self):

        return len(self.labels)

    

    def get_labels(self):

        return np.array(self.labels)
train_df_, valid_df_ = train_test_split(train_df, random_state=42, stratify=train_df.landmark_id.values)
train_ds = ImgDataset(train_df_)

valid_ds = ImgDataset(valid_df_)

sampler = BalanceBatchSampler(labels=train_ds.get_labels(), p=10, k=20)

train_dl = DataLoader(train_ds, sampler=sampler, batch_size=sampler.batch_size)

valid_dl = DataLoader(valid_ds, sampler=sampler, batch_size=sampler.batch_size)

loaders = {"train": train_dl, "valid": valid_dl}
from torchvision import models
model = models.resnext50_32x4d(pretrained=True)

for param in model.parameters():

    param.requires_grad = False

    

head = nn.Sequential(

    nn.Linear(1000, 512),

    nn.ReLU(),

    nn.Linear(512, 100),

)

model = nn.Sequential(

    model,

    head,

)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
import wandb



#wandb.login("never", "")

wandb.init(project="landmarks")
sampler_inbatch = data.HardTripletsSampler(norm_required=False)

criterion = TripletMarginLossWithSampler(margin=0.5, sampler_inbatch=sampler_inbatch)



# 4. training with catalyst Runner

callbacks = [

    dl.ControlFlowCallback(dl.CriterionCallback(), loaders="train"),

    dl.WandbLogger(log_on_batch_end=True, project="landmarks"),

]



runner = dl.SupervisedRunner(device=utils.get_device())

#runner.train(

#    model=model,

#    criterion=criterion,

#    optimizer=optimizer,

#    callbacks=callbacks,

#    loaders=loaders,

#    minimize_metric=False,

#    verbose=True,

#    num_epochs=200,

#)   