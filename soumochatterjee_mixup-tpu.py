!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
import os
import torch
import pandas as pd
from scipy import stats
import numpy as np

from collections import OrderedDict, namedtuple
import torch.nn as nn
from torch.optim import lr_scheduler
import joblib

import logging
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
import sys
from sklearn import metrics, model_selection

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
import glob

train_files = glob.glob('/kaggle/input/tpu-getting-started/*/train/*.tfrec')
val_files = glob.glob('/kaggle/input/tpu-getting-started/*/val/*.tfrec')
test_files = glob.glob('/kaggle/input/tpu-getting-started/*/test/*.tfrec')
import tensorflow as tf

# Create a dictionary describing the features.
train_feature_description = {
    'class': tf.io.FixedLenFeature([], tf.int64),
    'id': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, train_feature_description)

train_ids = []
train_class = []
train_images = []

for i in train_files:
  train_image_dataset = tf.data.TFRecordDataset(i)

  train_image_dataset = train_image_dataset.map(_parse_image_function)

  ids = [str(id_features['id'].numpy())[2:-1] for id_features in train_image_dataset] # [2:-1] is done to remove b' from 1st and 'from last in train id names
  train_ids = train_ids + ids

  classes = [int(class_features['class'].numpy()) for class_features in train_image_dataset]
  train_class = train_class + classes

  images = [image_features['image'].numpy() for image_features in train_image_dataset]
  train_images = train_images + images
val_feature_description = {
    'class': tf.io.FixedLenFeature([], tf.int64),
    'id': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, val_feature_description)

val_ids = []
val_class = []
val_images = []

for i in val_files:
    val_image_dataset = tf.data.TFRecordDataset(i)

    val_image_dataset = val_image_dataset.map(_parse_image_function)

    ids = [str(image_features['id'].numpy())[2:-1] for image_features in val_image_dataset]
    val_ids += ids

    classes = [int(image_features['class'].numpy()) for image_features in val_image_dataset]
    val_class += classes 

    images = [image_features['image'].numpy() for image_features in val_image_dataset]
    val_images += images
import IPython.display as display

display.display(display.Image(data=val_images[10000]))
from PIL import Image
import cv2
import albumentations
import torch
import numpy as np
import io
from torch.utils.data import Dataset

# Making the dataset class for training and testing Flower images

class FlowerDataset(Dataset):
    def __init__(self, id , classes , image , img_height , img_width, mean , std , is_valid):
        self.id = id
        self.classes = classes
        self.image = image
        self.is_valid = is_valid
        if self.is_valid == 1: # transforms for validation images
            self.aug = albumentations.Compose([
               albumentations.Resize(img_height , img_width, always_apply = True) ,
               albumentations.Normalize(mean , std , always_apply = True) 
            ])
        else:                  # transfoms for training images 
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height , img_width, always_apply = True) ,
                albumentations.Normalize(mean , std , always_apply = True),
                albumentations.ShiftScaleRotate(shift_limit = 0.0625,
                                                scale_limit = 0.1 ,
                                                rotate_limit = 5,
                                                p = 0.9)
            ]) 
        
    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, index):
        id = self.id[index]
        img = np.array(Image.open(io.BytesIO(self.image[index])))  # converting byte format of images to numpy array
        img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        img = self.aug(image = img)['image']
        img = np.transpose(img , (2,0,1)).astype(np.float32) # 2,0,1 because pytorch excepts image channel first then dimension of image
       
        return torch.tensor(img, dtype = torch.float),torch.tensor(self.classes[index], dtype = torch.long)
        
# creating object for the dataset class 
train_dataset = FlowerDataset(id = train_ids, classes = train_class, image = train_images, 
                        img_height = 224 , img_width = 224, 
                        mean = (0.485, 0.456, 0.406),
                        std = (0.229, 0.224, 0.225) , is_valid = 0)

val_dataset = FlowerDataset(id = val_ids, classes = val_class, image = val_images, 
                        img_height = 224 , img_width = 224, 
                        mean = (0.485, 0.456, 0.406),
                        std = (0.229, 0.224, 0.225) , is_valid = 1)
import matplotlib.pyplot as plt
%matplotlib inline

idx = 10000 # taking index for 10000th image out of 51000 images
img = val_dataset[idx][0]

print(val_dataset[idx][1]) # val_dataset label is one Hot encoded

npimg = img.numpy()
plt.imshow(np.transpose(npimg, (1,2,0)))
train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)

valid_sampler = torch.utils.data.distributed.DistributedSampler(
          val_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False)
TRAIN_BATCH_SIZE = 128

from torch.utils.data import DataLoader

training_dataloader = DataLoader(train_dataset,
                        num_workers=4,
                        batch_size=TRAIN_BATCH_SIZE,
                        sampler=train_sampler,
                        drop_last=True
                       )

val_dataloader = DataLoader(val_dataset,
                        num_workers=4,
                        batch_size=TRAIN_BATCH_SIZE,
                        sampler=valid_sampler,
                        drop_last=False
                       )
device = xm.xla_device()

!pip install efficientnet_pytorch

import efficientnet_pytorch

model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
import torch.nn as nn
import torch.nn.functional as F

# increasing few layers in our model
class EfficientNet_b0(nn.Module):
    def __init__(self):
        super(EfficientNet_b0, self).__init__()
        self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
        
        self.dense_layer_1 = nn.Linear(1280 , 512)
        self.batchNorm_layer = nn.BatchNorm1d(512)
        self.droput_layer = nn.Dropout(0.2)
        
        self.dense_layer_2 = nn.Linear(512,216)
        
        self.dense_layer_3 = nn.Linear(216,104)
        
        
    def forward(self, inputs):
        x = self.model.extract_features(inputs)

        # Pooling and final linear layer
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        
        x = self.dense_layer_1(x)
        x = self.batchNorm_layer(x)
        x = self.droput_layer(x)
        
        x = self.dense_layer_2(x)
        x = torch.relu(x)
        
        x = self.dense_layer_3(x)
        x = torch.log_softmax(x , dim =1)
        return x
    
model = EfficientNet_b0()
model = model.to(device)
model
# installing torchcontrib for Stochastic Weight Averaging in PyTorch 
!pip install torchcontrib
#for Stochastic Weight Averaging in PyTorch
from torchcontrib.optim import SWA

EPOCHS = 25
num_train_steps = int(len(train_dataset) / TRAIN_BATCH_SIZE / xm.xrt_world_size() * EPOCHS)

# printing the no of training steps for each epoch of our training dataloader  
xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')



params = list(model.dense_layer_1.parameters()) + list(model.dense_layer_2.parameters()) + list(model.dense_layer_3.parameters())

base_optimizer = torch.optim.Adam(params, lr=1e-4* xm.xrt_world_size())



optimizer = SWA(base_optimizer, swa_start=5, swa_freq=5, swa_lr=0.05)

loss_fn = torch.nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
!pip install torchtoolbox

from torchtoolbox.tools import mixup_data, mixup_criterion

# defining the training loop
def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):
    running_loss = 0.0
    model.train()
    
    alpha = 0.2
    for i, (inputs,labels) in enumerate(data_loader):
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        
        inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = mixup_criterion(loss_fn, outputs, labels_a, labels_b, lam)

        loss.backward()
        xm.optimizer_step(optimizer)

        running_loss += loss.item() * inputs.size(0)

    if scheduler is not None:
        scheduler.step()
            
    train_loss = running_loss / float(len(train_dataset))
    xm.master_print('training Loss: {:.4f}'.format(train_loss))
def eval_loop_fn(data_loader, model, device):
    running_loss = 0.0
    running_corrects = 0.0
    model.eval()
    
    for inputs,labels in data_loader:
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = loss_fn(outputs, labels)
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    valid_loss = running_loss / float(len(val_dataset))
    epoch_acc = running_corrects / float(len(val_dataset))
    xm.master_print('validation Loss: {:.4f} Acc: {:.4f}'.format(valid_loss, epoch_acc))
# training the model in _run function
def _run():
    for param in model.parameters():
        param.requires_grad = False
    
    for param in params:
        param.requires_grad = True
    
    for epoch in range(EPOCHS):
        xm.master_print(f"Epoch --> {epoch+1} / {EPOCHS}")
        xm.master_print(f"-------------------------------")
        para_loader = pl.ParallelLoader(training_dataloader, [device])
        train_loop_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler=scheduler)

        para_loader = pl.ParallelLoader(val_dataloader, [device])
        eval_loop_fn(para_loader.per_device_loader(device), model, device)
def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    a = _run()
    optimizer.swap_swa_sgd()
    
# applying multiprocessing so that images get paralley trained in different cores of kaggle-tpu
FLAGS={}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=1, start_method='fork')
