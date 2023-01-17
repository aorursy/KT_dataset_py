%matplotlib inline
from typing import List

import logging

from typing import Optional

from functools import partial

from typing import Tuple

from typing import Union





import torch.nn as nn

import numpy as np

import os

import pandas as pd

import torch

from torch.optim import Adam

from torchvision.models.resnet import BasicBlock

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

from PIL import Image

from matplotlib import pyplot as plt

from torchvision.models.resnet import ResNet

from sklearn.metrics import roc_auc_score

from torch import Tensor

from torchvision import transforms

from torch.autograd import Variable
DATA_FOLDER = '../input'

LABELS = f'{DATA_FOLDER}/train_labels.csv'

TRAIN_IMAGES_FOLDER = f'{DATA_FOLDER}/train'

USE_GPU = torch.cuda.is_available()
logging.basicConfig(level='INFO')

logger = logging.getLogger()
def read_labels(path_to_file: str) -> pd.DataFrame:

    labels = pd.read_csv(path_to_file)

    return labels





def format_labels_for_dataset(labels: pd.DataFrame) -> np.array:

    return labels['label'].values.reshape(-1, 1)





def format_path_to_images_for_dataset(labels: pd.DataFrame, path: str) -> List:

    return [os.path.join(path, f'{f}.tif') for f in labels['id'].values]





def train_valid_split(df: pd.DataFrame) -> Tuple:

    limit_df = 50000

    df = df.sample(n = df.shape[0])

    df = df.iloc[:limit_df]

    split = 40000

    train = df.iloc[:split]

    valid = df.iloc[:split]

    return train, valid
class MainDataset(Dataset):

    def __init__(self,

                 x_dataset: Dataset,

                 y_dataset: Dataset,

                 x_tfms: Optional = None):

        self.x_dataset = x_dataset

        self.y_dataset = y_dataset

        self.x_tfms = x_tfms



    def __len__(self) -> int:

        return self.x_dataset.__len__()



    def __getitem__(self, index: int) -> Tuple:

        x = self.x_dataset[index]

        y = self.y_dataset[index]

        if self.x_tfms is not None:

            x = self.x_tfms(x)

        return x, y

    

class ImageDataset(Dataset):

    def __init__(self, paths_to_imgs: List):

        self.paths_to_imgs = paths_to_imgs



    def __len__(self) -> int:

        return len(self.paths_to_imgs)



    def __getitem__(self, index: int) -> Image.Image:

        img = Image.open(self.paths_to_imgs[index])

        return img





class LabelDataset(Dataset):

    def __init__(self, labels: List):

        self.labels = labels



    def __len__(self) -> int:

        return len(self.labels)



    def __getitem__(self, index: int) -> int:

        return self.labels[index]
labels = read_labels(LABELS)

train, valid = train_valid_split(labels)



train_labels = format_labels_for_dataset(train)

valid_labels = format_labels_for_dataset(valid)



train_images = format_path_to_images_for_dataset(train, TRAIN_IMAGES_FOLDER)

valid_images = format_path_to_images_for_dataset(valid, TRAIN_IMAGES_FOLDER)



train_images_dataset = ImageDataset(train_images)

valid_images_dataset = ImageDataset(valid_images)

train_labels_dataset = LabelDataset(train_labels)

valid_labels_dataset = LabelDataset(valid_labels)
x_tfms = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]

    )

])
train_dataset = MainDataset(train_images_dataset, train_labels_dataset, x_tfms)

valid_dataset = MainDataset(valid_images_dataset, valid_labels_dataset, x_tfms)
shuffle = True

batch_size = 512

num_workers = 0

train_dataloader = DataLoader(train_dataset, 

                              batch_size=batch_size, 

                              shuffle=shuffle, 

                              num_workers=num_workers)

valid_dataloader = DataLoader(valid_dataset, 

                              batch_size=batch_size, 

                              shuffle=False, 

                              num_workers=num_workers)
# we will create custom resnet with 9 layers so it can process 96x96 images

def to_gpu(tensor):

    return tensor.cuda() if USE_GPU else tensor





def create_resnet9_model(output_dim: int = 1) -> nn.Module:

    model = ResNet(BasicBlock, [1, 1, 1, 1])

    in_features = model.fc.in_features

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    model.fc = nn.Linear(in_features, output_dim)

    model = to_gpu(model)

    return model
resnet9 = create_resnet9_model(output_dim=1)

resnet9
lr = 1e-3

optimizer = Adam(resnet9.parameters(), lr=lr)
loss = nn.BCEWithLogitsLoss()
def train_one_epoch(model,

                    train_dataloader,

                    valid_dataloader,

                    loss,

                    optimizer,

                    loss_writer_train,

                    loss_writer_valid,

                    do_step_trigger,

                    valid_loss_trigger,

                    train_loss_trigger,

                    ):

    model.train()

    y_true_train, y_hat_train = [], []

    for iteration, (x, y) in enumerate(train_dataloader):

        x = Variable(T(x), requires_grad=True)

        y = Variable(T(y), requires_grad=True)

        output = model(x)

        y_true_train.append(to_numpy(y))

        y_hat_train.append(to_numpy(output))

        loss_values = loss(output, y)

        loss_values.backward()

        if do_step_trigger(iteration):

            optimizer.step()

            optimizer.zero_grad()

        if train_loss_trigger(iteration):

            auc_writer(y_true_train, y_hat_train, iteration)

            y_true_train, y_hat_train = [], []

        if valid_loss_trigger(iteration):

            y_true, y_hat = predict(model, valid_dataloader)

            auc_writer(y_true, y_hat, iteration)

    return model
def predict(model, dataloader):

    model.eval()

    y_true, y_hat = [], []

    with torch.no_grad():

        for x, y in dataloader:

            x = Variable(T(x))

            y = Variable(T(y))

            output = model(x)

            y_true.append(to_numpy(y))

            y_hat.append(to_numpy(output))

    return y_true, y_hat
def iteration_trigger(iteration, every_x_iterations):

    if every_x_iterations == 1:

        return True

    elif iteration > 0 and iteration % every_x_iterations == 0:

        return True

    else:

        return False





def init_triggers(step=1, valid=10, train=10):

    do_step_trigger = partial(iteration_trigger, every_x_iterations=step)

    valid_loss_trigger = partial(iteration_trigger, every_x_iterations=valid)

    train_loss_trigger = partial(iteration_trigger, every_x_iterations=train)

    return do_step_trigger, valid_loss_trigger, train_loss_trigger
def auc_writer(y_true, y_hat, iteration):

    try:

        auc = roc_auc_score(np.vstack(y_true), np.vstack(y_hat))

    except:

        auc = -1

    logger.info(f'iteration: {iteration}, auc: {auc}')
def T(tensor):

    if not torch.is_tensor(tensor):

        tensor = torch.FloatTensor(tensor)

    else:

        tensor = tensor.type(torch.FloatTensor)

    if USE_GPU:

        tensor = to_gpu(tensor)

    return tensor





def to_numpy(tensor: Union[Tensor, Image.Image, np.array]) -> np.ndarray:

    if type(tensor) == np.array or type(tensor) == np.ndarray:

        return np.array(tensor)

    elif type(tensor) == Image.Image:

        return np.array(tensor)

    elif type(tensor) == Tensor:

        return tensor.cpu().detach().numpy()

    else:

        raise ValueError(msg)
loss_writer_train = auc_writer

loss_writer_valid = auc_writer



# backpropagate - every iter

# calculate AUC on valid - every 20 iters 

# AUC on train - every 10 iters

do_step_trigger, valid_loss_trigger, train_loss_trigger = init_triggers(1, 20, 10)
resnet9 = train_one_epoch(

    resnet9,

    train_dataloader,

    valid_dataloader,

    loss,

    optimizer,

    loss_writer_train,

    loss_writer_valid,

    do_step_trigger,

    valid_loss_trigger,

    train_loss_trigger,

)
y_true, y_hat = predict(resnet9, valid_dataloader)
predictions = pd.DataFrame(

    list(

        zip(

            valid_labels.reshape(-1), 

            np.vstack(y_hat).reshape(-1),

            valid_images

        )

    ), 

     columns=['true', 'pred', 'files'])

predictions.head(3)
max_charts = 60

def implot(files, w=2, h=2, cols=12):

    rows = len(files) / cols + 1

    images = [Image.open(f) for f in files]

    plt.figure(figsize = (cols * w, rows * h))

    plt.tight_layout()

    for chart, img in enumerate(images, 1):

        ax = plt.subplot(rows, cols, chart)

        ax.imshow(np.array(img))

        ax.axis('off')
files = predictions[predictions['true']==1].sort_values('pred')['files'].values[:max_charts]

implot(files)
files = predictions[predictions['true']==0].sort_values('pred', ascending=False)['files'].values[:max_charts]

implot(files)
files = predictions[predictions['true']==1].sort_values('pred', ascending=False)['files'].values[:max_charts]

implot(files)
files = predictions[predictions['true']==0].sort_values('pred', ascending=True)['files'].values[:max_charts]

implot(files)