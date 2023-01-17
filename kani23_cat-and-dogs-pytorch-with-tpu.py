#https://www.kaggle.com/dhananjay3/pytorch-xla-for-tpu-with-multiprocessing

# https://www.kaggle.com/iavinas/cat-vs-dog-transfer-learning
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py > /dev/null

!python pytorch-xla-env-setup.py --version 20200420 --apt-packages libomp5 libopenblas-dev > /dev/null
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import time



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import torch_xla

import torch_xla.debug.metrics as metrics

import torch_xla.distributed.data_parallel as dp

import torch_xla.distributed.parallel_loader as pl

import torch_xla.utils.utils as xu

import torch_xla.core.xla_model as xm

import torch_xla.distributed.xla_multiprocessing as xmp

import torch_xla.test.test_utils as test_utils





import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

from torchvision import transforms, datasets, models

from torch.optim import lr_scheduler



from sklearn.model_selection import train_test_split



from tqdm.notebook import tqdm

import gc

import os

import random



import cv2





from contextlib import contextmanager

from pathlib import Path

from collections import defaultdict, Counter
#from : https://www.kaggle.com/yasufuminakama/panda-se-resnext50-classification-baseline/data

# ====================================================

# Utils

# ====================================================



@contextmanager

def timer(name):

    t0 = time.time()

    LOGGER.info(f'[{name}] start')

    yield

    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')



    

def init_logger(log_file='train.log'):

    from logging import getLogger, DEBUG, FileHandler,  Formatter,  StreamHandler

    

    log_format = '%(asctime)s %(levelname)s %(message)s'

    

    stream_handler = StreamHandler()

    stream_handler.setLevel(DEBUG)

    stream_handler.setFormatter(Formatter(log_format))

    

    file_handler = FileHandler(log_file)

    file_handler.setFormatter(Formatter(log_format))

    

    logger = getLogger('catanddog')

    logger.setLevel(DEBUG)

    logger.addHandler(stream_handler)

    logger.addHandler(file_handler)

    

    return logger



LOG_FILE = 'train.log'

LOGGER = init_logger(LOG_FILE)





def seed_torch(seed=42):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



seed_torch(seed=42)
DATA_DIR = "/kaggle/input/cat-and-dog/"
train_data_dir = DATA_DIR + "/training_set/training_set"

transform = transforms.Compose([

    transforms.Resize([256, 256]),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                  std=[0.229, 0.224, 0.225]),

    

])

train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)

test_data_dir = DATA_DIR + "/test_set/test_set"

transform = transforms.Compose([

    transforms.Resize([256, 256]),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                  std=[0.229, 0.224, 0.225]),

])

valid_dataset = datasets.ImageFolder(test_data_dir, transform=transform)

from PIL import Image

import matplotlib.pyplot as plt 



tl = torch.utils.data.DataLoader(train_dataset, batch_size=4,) 

dataiter = iter(tl)

images, labels = dataiter.next()

images = images.numpy()

classes = ['cat', 'dog']

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(4):

    ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])

    plt.imshow(np.transpose(images[idx], (1, 2, 0)))

    ax.set_title(classes[labels[idx]])
class densenet121(nn.Module):



    def __init__(self):

        super(densenet121, self).__init__()

        

        self.base_model = models.densenet121(pretrained=True)

        self.fc = nn.Linear(1024, 2)

        self.output = nn.LogSoftmax(dim=1)

        

    def forward(self, x):

        bs, _, _, _ = x.shape

        x = self.base_model.features(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

        x = self.fc(x)

        x = self.output(x)

        

        return x
# test 

model = densenet121()

x = np.random.rand(1, 3, 224, 224)

out = model(torch.tensor(x, dtype=torch.float))

print(out.shape)
NUM_EPOCH = 10

BATCH_SIZE = 16



def train_model():

    global train_dataset, valid_dataset

    

    torch.manual_seed(42)

    

    train_sampler = torch.utils.data.distributed.DistributedSampler(

        train_dataset,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        shuffle=True)

    

    train_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        sampler=train_sampler,

        num_workers=0,

        drop_last=True) # print(len(train_loader))

    

    valid_sampler = torch.utils.data.distributed.DistributedSampler(

        valid_dataset,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        )

        

    valid_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=BATCH_SIZE ,

        sampler=valid_sampler,

        shuffle=False,

        num_workers=0,

        drop_last=True)

    

    xm.master_print(f"Train for {len(train_loader)} steps per epoch")

    # Scale learning rate to num cores

    lr  = 0.001 * xm.xrt_world_size()



    # Get loss function, optimizer, and model

    device = xm.xla_device()

    

    global model

    

    model = model.to(device)



    criterion = nn.NLLLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    scheduler = lr_scheduler.OneCycleLR(optimizer, 

                           lr, 

                           div_factor=10.0, 

                           final_div_factor=50.0, 

                           epochs=NUM_EPOCH,

                           steps_per_epoch=len(train_loader))

    

    

    

    def train_loop_fn(loader):

        tracker = xm.RateTracker()

        model.train()

        running_loss = 0.0

        tk0 = tqdm(loader, total=int(len(train_loader)))

        for x, (data, target) in enumerate(tk0):

            data = data.to(device, dtype=torch.float)

            target = target.to(device, dtype=torch.float)

            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, target)

            loss.backward()

            

            xm.optimizer_step(optimizer)

            running_loss += loss.item()

            tk0.set_postfix(loss=(running_loss))

        epoch_loss = running_loss / len(train_loader)

        print('Training Loss: {:.8f}'.format(epoch_loss))

            



            

                

    def test_loop_fn(loader):

        with torch.no_grad():

            total_samples, correct = 0, 0

            model.eval()

            tk0 = tqdm(loader, total=int(len(valid_loader)))

            

            for data, target in loader:

                data = data.to(device, dtype=torch.float)

                target = target.to(device, dtype=torch.float)

                output = model(data)

                pred = output.max(1, keepdim=True)[1]

                correct += pred.eq(target.view_as(pred)).sum().item()

                total_samples += data.size()[0]

            accuracy = 100.0 * correct / total_samples

            print('[xla:{}] Accuracy={:.2f}%'.format(xm.get_ordinal(), accuracy), flush=True)

            model.train()

        return accuracy

    



    accuracy = []

    for epoch in range(1, NUM_EPOCH + 1):

        start = time.time()

        para_loader = pl.ParallelLoader(train_loader, [device])

        train_loop_fn(para_loader.per_device_loader(device))

        

        para_loader = pl.ParallelLoader(valid_loader, [device])

        accuracy.append(test_loop_fn(para_loader.per_device_loader(device)))

        xm.master_print("Finished training epoch {} acc {:.2f} in {:.2f} sec"\

                        .format(epoch, accuracy[-1], time.time() - start))        

        xm.save(model.state_dict(), "./model.pt")

        

#         if epoch == 15: #unfreeze

#                 for param in model.base_model.parameters():

#                     param.requires_grad = True



    return accuracy

def _mp_fn(rank, flags):

    global acc_list

    torch.set_default_tensor_type('torch.FloatTensor')

    res = train_model()



FLAGS={}

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')