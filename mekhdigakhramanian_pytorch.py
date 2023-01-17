!mkdir -p /tmp/pip/cache/

!cp ../input/resources-for-google-landmark-recognition-2020/efficientnet_pytorch-0.6.3-py3-none-any.whl /tmp/pip/cache/

!pip install --no-index --find-links /tmp/pip/cache/ efficientnet_pytorch
import os

import gc

gc.enable()

import sys

import math

import json

import time

import random

from glob import glob

from datetime import datetime



import cv2

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

import multiprocessing

from sklearn.preprocessing import LabelEncoder



import torch

import torchvision

from torch import Tensor

from torchvision import transforms

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

from torch.nn.parameter import Parameter

from torch.optim import lr_scheduler, Adam

from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SequentialSampler

from tqdm import tqdm



import efficientnet_pytorch



import albumentations as A



import sklearn



import warnings

warnings.filterwarnings("ignore")
IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None

MIN_SAMPLES_PER_CLASS = 40

BATCH_SIZE = 64

NUM_WORKERS = multiprocessing.cpu_count()

MAX_STEPS_PER_EPOCH = 15000

NUM_EPOCHS = 50

LOG_FREQ = 10

NUM_TOP_PREDICTS = 1

ENABLE_FAST_SKIP = True
train = pd.read_csv('../input/landmark-recognition-2020/train.csv')

test = pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')

train_dir = '../input/landmark-recognition-2020/train/'

test_dir = '../input/landmark-recognition-2020/test/'
IMG_SIZE = 128



class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe: pd.DataFrame, image_dir:str, mode: str):

        self.df = dataframe

        self.mode = mode

        self.image_dir = image_dir

        

        transforms_list = []

        if self.mode == 'train':

            # Increase image size from (64,64) to higher resolution,

            # Make sure to change in RandomResizedCrop as well.

            transforms_list = [

                transforms.Resize((IMG_SIZE,IMG_SIZE)),

                transforms.RandomHorizontalFlip(),

                transforms.RandomChoice([

                    transforms.RandomResizedCrop(IMG_SIZE),

                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),

                    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2),

                                            scale=(0.8, 1.2), shear=15,

                                            resample=Image.BILINEAR)

                ]),

                transforms.ToTensor(),

                transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                      std=[0.229, 0.224, 0.225]),

            ]

        else:

            transforms_list.extend([

                # Keep this resize same as train

                transforms.Resize((IMG_SIZE,IMG_SIZE)),

                transforms.ToTensor(),

                transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                      std=[0.229, 0.224, 0.225]),

            ])

        self.transforms = transforms.Compose(transforms_list)



    def __getitem__(self, index: int):

        image_id = self.df.iloc[index].id

        image_path = f"{self.image_dir}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg"

        image = Image.open(image_path)

        image = self.transforms(image)



        if self.mode == 'test':

            return {'image':image}

        else:

            return {'image':image, 

                    'target':self.df.iloc[index].landmark_id}



    def __len__(self) -> int:

        return self.df.shape[0]
def load_data(train, test, train_dir, test_dir):

    counts = train.landmark_id.value_counts()

    selected_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index

    num_classes = selected_classes.shape[0]

    print('classes with at least N samples:', num_classes)



    train = train.loc[train.landmark_id.isin(selected_classes)]

    print('train_df', train.shape)

    print('test_df', test.shape)



    # filter non-existing test images

    exists = lambda img: os.path.exists(f'{test_dir}/{img[0]}/{img[1]}/{img[2]}/{img}.jpg')

    test = test.loc[test.id.apply(exists)]

    print('test_df after filtering', test.shape)



    label_encoder = LabelEncoder()

    label_encoder.fit(train.landmark_id.values)

    print('found classes', len(label_encoder.classes_))

    assert len(label_encoder.classes_) == num_classes



    train.landmark_id = label_encoder.transform(train.landmark_id)



    train_dataset = ImageDataset(train, train_dir, mode='train')

    test_dataset = ImageDataset(test, test_dir, mode='test')



    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,

                              shuffle=False, num_workers=4, drop_last=True)



    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,

                             shuffle=False, num_workers=NUM_WORKERS)



    return train_loader, test_loader, label_encoder, num_classes
def adam(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

    if isinstance(betas, str):

        betas = eval(betas)

    return Adam(parameters,

                lr=lr,

                betas=betas,

                eps=eps,

                weight_decay=weight_decay)
class AverageMeter:

    ''' Computes and stores the average and current value '''

    def __init__(self) -> None:

        self.reset()



    def reset(self) -> None:

        self.val = 0.0

        self.avg = 0.0

        self.sum = 0.0

        self.count = 0



    def update(self, val: float, n: int = 1) -> None:

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count
def GAP(predicts: torch.Tensor, confs: torch.Tensor, targets: torch.Tensor) -> float:

    ''' Simplified GAP@1 metric: only one prediction per sample is supported '''

    assert len(predicts.shape) == 1

    assert len(confs.shape) == 1

    assert len(targets.shape) == 1

    assert predicts.shape == confs.shape and confs.shape == targets.shape



    _, indices = torch.sort(confs, descending=True)



    confs = confs.cpu().numpy()

    predicts = predicts[indices].cpu().numpy()

    targets = targets[indices].cpu().numpy()



    res, true_pos = 0.0, 0



    for i, (c, p, t) in enumerate(zip(confs, predicts, targets)):

        rel = int(p == t)

        true_pos += rel



        res += true_pos / (i + 1) * rel



    res /= targets.shape[0] # FIXME: incorrect, not all test images depict landmarks

    return res
class EfficientNetEncoderHead(nn.Module):

    def __init__(self, depth, num_classes):

        super(EfficientNetEncoderHead, self).__init__()

        self.depth = depth

        self.base = efficientnet_pytorch.EfficientNet.from_name(f'efficientnet-b{self.depth}')

        pretrained_file = glob(f'../input/resources-for-google-landmark-recognition-2020/efficientnet-b{self.depth}*')[0]

        self.base.load_state_dict(torch.load(pretrained_file))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.output_filter = self.base._fc.in_features

        self.classifier = nn.Linear(self.output_filter, num_classes)

    def forward(self, x):

        x = self.base.extract_features(x)

        x = self.avg_pool(x).squeeze(-1).squeeze(-1)

        x = self.classifier(x)

        return x
def train_step(train_loader, 

          model, 

          criterion, 

          optimizer,

          epoch, 

          lr_scheduler):

    print(f'epoch {epoch}')

    batch_time = AverageMeter()

    losses = AverageMeter()

    avg_score = AverageMeter()



    model.train()

    num_steps = min(len(train_loader), MAX_STEPS_PER_EPOCH)



    print(f'total batches: {num_steps}')



    end = time.time()

    lr = None



    for i, data in enumerate(train_loader):

        input_ = data['image']

        target = data['target']

        batch_size, _, _, _ = input_.shape

        

        output = model(input_.cuda())

        loss = criterion(output, target.cuda())

        confs, predicts = torch.max(output.detach(), dim=1)

        avg_score.update(GAP(predicts, confs, target))

        losses.update(loss.data.item(), input_.size(0))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]['lr']

        

        batch_time.update(time.time() - end)

        end = time.time()



        if i % LOG_FREQ == 0:

            print(f'{epoch} [{i}/{num_steps}]\t'

                    f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'

                    f'loss {losses.val:.4f} ({losses.avg:.4f})\t'

                    f'GAP {avg_score.val:.4f} ({avg_score.avg:.4f})'

                    + str(lr))



    print(f' * average GAP on train {avg_score.avg:.4f}')
def inference(data_loader, model):

    model.eval()



    activation = nn.Softmax(dim=1)

    all_predicts, all_confs, all_targets = [], [], []



    with torch.no_grad():

        for i, data in enumerate(tqdm(data_loader, disable=IN_KERNEL)):

            if data_loader.dataset.mode != 'test':

                input_, target = data['image'], data['target']

            else:

                input_, target = data['image'], None



            output = model(input_.cuda())

            output = activation(output)



            confs, predicts = torch.topk(output, NUM_TOP_PREDICTS)

            all_confs.append(confs)

            all_predicts.append(predicts)



            if target is not None:

                all_targets.append(target)



    predicts = torch.cat(all_predicts)

    confs = torch.cat(all_confs)

    targets = torch.cat(all_targets) if len(all_targets) else None



    return predicts, confs, targets
def generate_submission(test_loader, model, label_encoder):

    sample_sub = pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')



    predicts_gpu, confs_gpu, _ = inference(test_loader, model)

    predicts, confs = predicts_gpu.cpu().numpy(), confs_gpu.cpu().numpy()



    labels = [label_encoder.inverse_transform(pred) for pred in predicts]

    print('labels')

    print(np.array(labels))

    print('confs')

    print(np.array(confs))



    sub = test_loader.dataset.df

    def concat(label: np.ndarray, conf: np.ndarray) -> str:

        return ' '.join([f'{L} {c}' for L, c in zip(label, conf)])

    sub['landmarks'] = [concat(label, conf) for label, conf in zip(labels, confs)]



    sample_sub = sample_sub.set_index('id')

    sub = sub.set_index('id')

    sample_sub.update(sub)



    sample_sub.to_csv('submission.csv')
if __name__ == '__main__':

    global_start_time = time.time()

    train_loader, test_loader, label_encoder, num_classes = load_data(train, test, train_dir, test_dir)



    if ENABLE_FAST_SKIP and test.id[0] == "00084cdf8f600d00":

        # This is a run on the public data, skip it to speed up submission run on private data.

        print("Skipping run on public test set.")

        sample_sub = pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')

        sample_sub.to_csv('submission.csv')

    else:

        model = EfficientNetEncoderHead(depth=4, num_classes=num_classes)

        model.cuda()

        

        criterion = nn.CrossEntropyLoss()



        optimizer = adam(model.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-3, weight_decay=1e-4)

        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*NUM_EPOCHS, eta_min=1e-6)



        for epoch in range(1, NUM_EPOCHS + 1):

            print('-' * 50)

            train_step(train_loader, model, criterion, optimizer, epoch, scheduler)



        print('inference mode')

        generate_submission(test_loader, model, label_encoder)