!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
import os
import time
import numpy as np 
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

!echo $TPU_NAME
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchvision.transforms as T 

# imports the torch_xla package
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
train_df = pd.read_csv(dirname+'/train.csv')
labels = train_df['label'].values
img_arrs = train_df.iloc[:,1:].values

idx = 10
img_arr = img_arrs[idx]/255.

plt.imshow(img_arr.reshape(28,28), cmap='gray')
plt.show()
transforms = T.Compose([T.ToPILImage(), T.Resize((224,224)), T.ToTensor()])
img_tensor = transforms(np.float32(img_arr.reshape(28,28,1)))
plt.imshow(img_tensor[0],cmap='gray')
plt.show()
class mnistDataset(Dataset):
    def __init__(self, img_arrs, labels = None):
        self.img_arrs = img_arrs
        self.labels = labels
        self.transforms = T.Compose([T.ToPILImage(), T.Resize((224,224)), T.ToTensor()])
        
    def __getitem__(self, idx):
        img_arr = self.img_arrs[idx]/255 
        original_img = img_arr.reshape(28,28,1)
        img_tensor = self.transforms(np.float32(original_img))
        if self.labels is None:
            return img_tensor
        else:
            target_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
            return img_tensor, target_tensor
    
    def __len__(self):
        return len(self.img_arrs)
dataset = mnistDataset(img_arrs,labels)
print(dataset.__len__())
batch_size = 32
train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)
trainiter = iter(train_loader)
img_tensor, target_tensor = next(trainiter)
print(img_tensor.size())
print(target_tensor.size())

fig = plt.figure(figsize=(25, 8))
plot_size = 32
for idx in np.arange(plot_size):
    ax = fig.add_subplot(4, plot_size/4, idx+1, xticks=[], yticks=[])
    ax.imshow(img_tensor[idx][0], cmap='gray')
net = torchvision.models.alexnet(num_classes=10)
layers = [nn.Conv2d(1, 64, kernel_size = 11, stride = 4, padding = 2)] + list(net.features.children())[1:]
net.features = nn.Sequential(*layers)
net
def map_fn(index, flags):
    
    ## Sets a common random seed - both for initialization and ensuring graph is the same
    torch.manual_seed(flags['seed'])
    
    ## Acquires the (unique) Cloud TPU core corresponding to this process's index
    device = xm.xla_device()   
    
    ## Dataloader construction

    train_dataset = mnistDataset(img_arrs,labels)
    
    # Creates the (distributed) sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    
    # Creates dataloaders, which load data in batches
    # Note: test loader is not shuffled or sampled (include in sampler)
    train_loader = DataLoader(train_dataset,
                              batch_size=flags['batch_size'],
                              sampler=train_sampler,
                              num_workers=flags['num_workers'],
                              drop_last=True)
    
    
    ## Network, optimizer, and loss function creation
    
    # create network Note: each process has its own identical copy of the model
    net = torchvision.models.alexnet(num_classes=10)
    layers = [nn.Conv2d(1, 64, kernel_size = 11, stride = 4, padding = 2)] + list(net.features.children())[1:]
    net.features = nn.Sequential(*layers)
    net = net.to(device).train()

    loss_window = deque(maxlen=flags['window'])
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    
    ## Trains
    tracker = xm.RateTracker()
    train_start = time.time()
    for epoch in range(flags['num_epochs']):
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        for x, (data, targets) in enumerate(para_train_loader):
            output = net(data)
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            loss_window.append(loss.item())

      # Note: optimizer_step uses the implicit Cloud TPU context to
      #  coordinate and synchronize gradient updates across processes.
      #  This means that each process's network has the same weights after
      #  this is called.
        
            xm.optimizer_step(optimizer)  # Note: barrier=True not needed when using ParallelLoader 
            tracker.add(flags['batch_size'])
            if x % flags['window']  == 0:
                print('[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
            xm.get_ordinal(), x, np.mean(loss_window), tracker.rate(),
            tracker.global_rate(), time.asctime()), flush=True)

    elapsed_train_time = time.time() - train_start
    print("Process", index, "finished training. Train time was:", elapsed_train_time) 

flags = {}
flags['batch_size'] = 32
flags['num_workers'] = 4
flags['num_epochs'] = 1
flags['seed'] = 1234
flags['window'] = 20

xmp.spawn(map_fn, args=(flags,), nprocs=8, start_method='fork')
def train_net():
    torch.manual_seed(FLAGS['seed'])
    device = xm.xla_device()   
    train_dataset = mnistDataset(img_arrs,labels)
    
    train_sampler = DistributedSampler(train_dataset,
                                       num_replicas = xm.xrt_world_size(),
                                       rank = xm.get_ordinal(),
                                       shuffle = True)
    
    train_loader = DataLoader(train_dataset,
                              batch_size = FLAGS['batch_size'],
                              sampler = train_sampler,
                              num_workers = FLAGS['num_workers'],
                              drop_last = True)
    
    ###
    model = torchvision.models.alexnet(num_classes=10)
    layers = [nn.Conv2d(1, 64, kernel_size = 11, stride = 4, padding = 2)] + list(net.features.children())[1:]
    model.features = nn.Sequential(*layers)
    model = net.to(device)
    ###

    loss_window = deque(maxlen = FLAGS['log_steps'])
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    
    def train_loop_fn(loader):
        tracker = xm.RateTracker()
        model.train()
        for x, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            loss_window.append(loss.item())
            xm.optimizer_step(optimizer)
            tracker.add(FLAGS['batch_size'])
            if (x+1) % FLAGS['log_steps'] == 0:
                print('[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
            xm.get_ordinal(), x+1, np.mean(loss_window), tracker.rate(),
            tracker.global_rate(), time.asctime()), flush=True)
    
    for epoch in range(1, FLAGS['num_epochs'] + 1):
        para_loader = pl.ParallelLoader(train_loader, [device])
        train_loop_fn(para_loader.per_device_loader(device))
        xm.master_print("Finished training epoch {}".format(epoch))
        
    
def _mp_fn(rank, flags):
    global FLAGS
    FLAGS = flags
    torch.set_default_tensor_type('torch.FloatTensor')
    train_start = time.time()
    train_net()
    elapsed_train_time = time.time() - train_start
    print("Process", rank, "finished training. Train time was:", elapsed_train_time)


# Define Parameters
FLAGS = {}
FLAGS['seed'] = 1
FLAGS['batch_size'] = 128
FLAGS['num_workers'] = 4
FLAGS['num_epochs'] = 2
FLAGS['num_cores'] = 8
FLAGS['log_steps'] = 20
xmp.spawn(_mp_fn, args = (FLAGS,), nprocs = FLAGS['num_cores'],start_method='fork')
try:
    device = xm.xla_device()
    is_tpu = True
except:
    is_tpu = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
net = torchvision.models.alexnet(num_classes=10)
layers = [nn.Conv2d(1, 64, kernel_size = 11, stride = 4, padding = 2)] + list(net.features.children())[1:]
net.features = nn.Sequential(*layers)
net = net.to(device).train()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())
num_epochs = 1
window = 100
loss_window = deque(maxlen=window)

start_time = time.time()

for epoch in range(num_epochs):
    for counter, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        output = net(data)

        loss = criterion(output, targets)
        loss_window.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if is_tpu: 
            xm.optimizer_step(optimizer, barrier=True)  # Note: Cloud TPU-specific code!
        else:
            optimizer.step()
            
        print('\rEpoch {}/{}\tAverage Score: {:.2f}'.format(epoch,counter*batch_size, np.mean(loss_window)),end="")
        if counter % window == 0:
            print('\rEpoch {}/{}\tAverage Score: {:.2f}'.format(epoch,counter*batch_size, np.mean(loss_window)))

elapsed_time = time.time() - start_time
print ("\t Spent ", elapsed_time, " seconds training for ", num_epochs, " epoch(s) on a single core.")