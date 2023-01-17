# for TPU
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
!pip install efficientnet_pytorch > /dev/null
!pip install albumentations > /dev/null
import warnings
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
import warnings

warnings.filterwarnings("ignore")
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import operator
from PIL import Image 
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from torchvision.transforms import ToTensor, RandomHorizontalFlip, Resize
from efficientnet_pytorch import EfficientNet
from transformers import AdamW, get_cosine_schedule_with_warmup
from albumentations import *
from albumentations.pytorch import ToTensor
from tqdm import tqdm
import json
import time
BASE_DIR = '../input/plant-pathology-2020-fgvc7/'
train_df = pd.read_csv(BASE_DIR +'train.csv')
train_df.head()
train_df['image_id'] = BASE_DIR + 'images/' + train_df['image_id'] + '.jpg'
train_df['label'] = [np.argmax(label) for label in train_df[['healthy','multiple_diseases','rust','scab']].values]
train_df.head()
class SimpleDataset(Dataset):
    def __init__(self, image_ids_df, labels_df, transform=None):
        self.image_ids = image_ids_df
        self.labels = labels_df
        self.transform = transform
        
    def __getitem__(self, idx):
        image = cv2.imread(self.image_ids.values[idx])
        label = self.labels.values[idx]
        
        sample = {
            'image': image,
            'label': label
        }
        
        if self.transform:
            sample = self.transform(**sample)
        
        image, label = sample['image'], sample['label']
        
        return image, label
    
    def __len__(self):
        return len(self.image_ids)
    

        
image_ids = train_df['image_id']
labels = train_df['label']
X_train, X_test, y_train, y_test = train_test_split(image_ids, labels, test_size=0.25, random_state=42)
train_transform = Compose(
    [
        Resize(224, 224),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
#         ShiftScaleRotate(rotate_limit=25.0, p=0.7),
#         OneOf(
#             [
#                 IAAEmboss(p=1),
#                 IAASharpen(p=1),
#                 Blur(p=1)
#             ], 
#             p=0.5
#         ),
#         IAAPiecewiseAffine(p=0.5),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), always_apply=True),
        ToTensor()
    ]
)
model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=4)
def _run(model):
     
    def train_fn(epoch, train_dataloader, optimizer, criterion, scheduler, device):

        running_loss = 0
        total = 0
        model.train()

        for batch_idx, (images, labels) in enumerate(train_dataloader, 1):

            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            xm.master_print(f'Batch: {batch_idx}, loss: {loss.item()}')

            loss.backward()
            xm.optimizer_step(optimizer)

            lr_scheduler.step()

    def valid_fn(epoch, valid_dataloader, criterion, device):

        running_loss = 0
        total = 0
        preds_acc = []
        labels_acc = []

        model.eval()

        for batch_idx, (images, labels) in enumerate(valid_dataloader, 1):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            
            xm.master_print(f'Batch: {batch_idx}, loss: {loss.item()}')

            running_loss += loss.item()
    
    
    EPOCHS = 20
    BATCH_SIZE = 64
    
    train_dataset = SimpleDataset(X_train, y_train, transform=train_transform)
    valid_dataset = SimpleDataset(X_test, y_test, transform=train_transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
          valid_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=1)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, sampler=valid_sampler, num_workers=1)
    
    device = xm.xla_device()
    model = model.to(device)
    
    lr = 0.4 * 1e-5 * xm.xrt_world_size()
    criterion = nn.CrossEntropyLoss()
    
    optimizer = AdamW(model.parameters(), lr=lr)
    num_train_steps = int(len(train_dataset) / BATCH_SIZE / xm.xrt_world_size() * EPOCHS)
    xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')
    num_train_steps = int(len(train_dataset) / BATCH_SIZE * EPOCHS)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )
    
    train_loss = []
    valid_loss = []
    best_loss = 1
    
    train_begin = time.time()
    for epoch in range(EPOCHS):
        
        para_loader = pl.ParallelLoader(train_dataloader, [device])

        start = time.time()
        print('*'*15)
        print(f'EPOCH: {epoch+1}')
        print('*'*15)

        print('Training.....')
        train_fn(epoch=epoch+1, 
                                  train_dataloader=para_loader.per_device_loader(device), 
                                  optimizer=optimizer, 
                                  criterion=criterion,
                                  scheduler=lr_scheduler,
                                  device=device)


        
        with torch.no_grad():
            
            para_loader = pl.ParallelLoader(valid_dataloader, [device])
            
            print('Validating....')
            valid_fn(epoch=epoch+1, 
                                      valid_dataloader=para_loader.per_device_loader(device), 
                                      criterion=criterion, 
                                      device=device)
            xm.save(
                model.state_dict(),
                f'efficientnet-b0-bs-8.pt'
            )
    
        print(f'Epoch completed in {(time.time() - start)/60} minutes')
    print(f'Training completed in {(time.time() - train_begin)/60} minutes')
# Start training processes
def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    a = _run(model)

FLAGS={}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
