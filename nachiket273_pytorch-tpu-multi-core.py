!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
%reload_ext autoreload

%autoreload 2

%matplotlib inline
import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import random

from PIL import Image

import time

from tqdm.notebook import tqdm



import torch_xla

import torch_xla.core.xla_model as xm

import torch_xla.debug.metrics as met

import torch_xla.distributed.parallel_loader as pl

import torch_xla.distributed.xla_multiprocessing as xmp

import torch_xla.utils.utils as xu



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, random_split, DataLoader



import torchvision.models as models

import torchvision.transforms as transforms

from torchvision.utils import make_grid
DATA_DIR = '../input/jovian-pytorch-z2g/Human protein atlas/'

TRAIN_DIR = DATA_DIR + "/" + "train"

TEST_DIR = DATA_DIR + "/" + "test"

TRAIN_CSV = DATA_DIR +"/" + "train.csv"
train_df = pd.read_csv(TRAIN_CSV)

train_df.head()

labels = {

  0: 'Mitochondria',

  1: 'Nuclear bodies',

  2: 'Nucleoli',

  3: 'Golgi apparatus',

  4: 'Nucleoplasm',

  5: 'Nucleoli fibrillar center',

  6: 'Cytosol',

  7: 'Plasma membrane',

  8: 'Centrosome',

  9: 'Nuclear speckles'

}
def encode_labels(label):

    target = torch.zeros(10)

    for l in str(label).split(' '):

        target[int(l)] = 1.

    return target
def decode_labels(target, thresh=0.5, return_label=False):

    result = []

    for i, tgt in enumerate(target):

        if tgt > thresh:

            if return_label:

                result.append(str(i) + ":" + labels[i] + "/")

            else:

                result.append(str(i))     

    return result
def set_seed(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
class AvgStats(object):

    def __init__(self):

        self.reset()



    def reset(self):

        self.losses =[]

        self.F1 =[]

        self.its = []



    def append(self, loss, F1, it):

        self.losses.append(loss)

        self.F1.append(F1)

        self.its.append(it)
def save_checkpoint(model, is_best, filename='data/checkpoint.pth'):

    """Save checkpoint if a new best is achieved"""

    if is_best:

        xm.save(model.state_dict(), filename)  # save checkpoint

    else:

        print ("=> Validation Accuracy did not improve")
def load_checkpoint(model, filename = 'data/checkpoint.pth'):

    sd = torch.load(filename, map_location=lambda storage, loc: storage)

    names = set(model.state_dict().keys())

    for n in list(sd.keys()):

        if n not in names and n+'_raw' in names:

            if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]

            del sd[n]

    model.load_state_dict(sd)
class ProteinDataset(nn.Module):

        def __init__(self, root_dir, label_df, transforms=None):

            assert(os.path.exists(root_dir))

            self.root_dir = root_dir

            self.label_df = label_df

            self.transforms = transforms



        def __len__(self):

            return len(self.label_df)



        def __getitem__(self, idx):

            row = self.label_df.loc[idx]

            img_id, label = row['Image'], row['Label']

            img = Image.open(self.root_dir + "/" + str(img_id) + ".png")

            if self.transforms:

                img = self.transforms(img)

            return img, encode_labels(label)
mean = [0.0793, 0.0530, 0.0545]

std = [0.1290, 0.0886, 0.1376]
normalize = transforms.Normalize(mean=mean, std=std)
train_stats = AvgStats()

test_stats = AvgStats()
def get_model():

    model = models.resnet34(pretrained=True)

    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)

    for param in model.parameters():

        param.require_grad = True

    return model
# Only instantiate model weights once in memory.

SERIAL_EXEC = xmp.MpSerialExecutor()

WRAPPED_MODEL = xmp.MpModelWrapper(get_model())
def fit(rank, flags):

    torch.set_default_tensor_type('torch.FloatTensor')

    

    bs = flags['bs']

    epochs = flags['epochs']

    seed = flags['seed']



    set_seed(seed)



    val_pct = 0.10



    msk = np.random.rand(len(train_df)) < (1- val_pct)



    train_split_df = train_df[msk].reset_index()

    val_split_df = train_df[~msk].reset_index()



    def F_score(output, label, threshold=0.5, beta=1, eps=1e-12):

        beta2 = beta**2



        y_pred = torch.ge(output.float(), threshold).float()

        y_true = label.float()



        true_positive = (y_pred * y_true).sum(dim=1)

        precision = true_positive.div(y_pred.sum(dim=1).add(eps))

        recall = true_positive.div(y_true.sum(dim=1).add(eps))



        return torch.mean(

            (precision*recall).

            div(precision.mul(beta2) + recall + eps).

            mul(1 + beta2))



    



    def get_dataset():

        train_tf = transforms.Compose([

            transforms.RandomCrop(512, padding=8, padding_mode='symmetric'),

            transforms.RandomHorizontalFlip(), 

            transforms.RandomRotation(10),

            transforms.ToTensor(),

            normalize

        ])



        valid_tf = transforms.Compose([

            transforms.RandomCrop(512, padding=8, padding_mode='symmetric'),

            transforms.ToTensor(),

            normalize

        ])



        train_ds = ProteinDataset(TRAIN_DIR, train_split_df, train_tf)

        valid_ds = ProteinDataset(TRAIN_DIR, val_split_df, valid_tf)

        return train_ds, valid_ds



    train_ds, valid_ds = SERIAL_EXEC.run(get_dataset)



    train_sampler = torch.utils.data.distributed.DistributedSampler(

        train_ds,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        shuffle=True

    )



    valid_sampler = torch.utils.data.distributed.DistributedSampler(

        valid_ds,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        shuffle=False)



    train_loader = DataLoader(train_ds, bs, sampler=train_sampler, num_workers=1, pin_memory=True)

    valid_loader = DataLoader(valid_ds, bs, sampler=valid_sampler, num_workers=1, pin_memory=True)



    device = xm.xla_device()

    model = WRAPPED_MODEL.to(device)



    criterion = nn.BCELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2*xm.xrt_world_size(), momentum=0.9, weight_decay=5e-4)



    def train(epoch, model, loader, optimizer, criterion):

        tracker = xm.RateTracker()

        model.train()

        running_loss = 0.

        running_F1 = 0.

        start_time = time.time()

        #t = tqdm(loader, leave=False, total=len(loader))



        for i, (ip, tgt) in enumerate(loader):

            optimizer.zero_grad()

            #ip, tgt = ip.to(device), tgt.to(device)                                    

            output = torch.sigmoid(model(ip))

            loss = criterion(output, tgt)

            running_loss += loss.item()

            

            # compute gradient and do SGD step

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            xm.optimizer_step(optimizer)



            # Append outputs

            running_F1 += F_score(output, tgt)



        trn_time = time.time() - start_time        

        trn_F1 = running_F1/len(loader)

        trn_losses = running_loss/len(loader)

        return trn_F1, trn_losses, trn_time



    def test(model, loader, criterion):

        with torch.no_grad():

            model.eval()

            running_loss = 0.

            running_F1 = 0.

            start_time = time.time()

            #t = tqdm(loader, leave=False, total=len(loader))



            for i, (ip, tgt) in enumerate(loader):

                #ip, tgt = ip.to(device), tgt.to(device)

                output = torch.sigmoid(model(ip))

                loss = criterion(output, tgt)

                running_loss += loss.item()

                running_F1 += F_score(output, tgt)



            val_time = time.time() - start_time

            F1_score = running_F1/len(loader)

            val_F1 = F1_score

            val_losses = running_loss/len(loader)

            return val_F1, val_losses, val_time



    best_F1 = 0

    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

    for j in range(1, epochs+1):

        para_loader = pl.ParallelLoader(train_loader, [device])

        trn_F1, trn_losses, trn_time = train(j, model, para_loader.per_device_loader(device), optimizer, criterion)

        train_stats.append(trn_losses, trn_F1, trn_time)

        para_loader = pl.ParallelLoader(valid_loader, [device])

        val_F1, val_losses, val_time = test(model, para_loader.per_device_loader(device), criterion)

        test_stats.append(val_losses, val_F1, val_time)

        if val_F1 > best_F1:

            save_checkpoint(model, True, './best_model.pth')

        sched.step()

        print("Epoch::{}, Trn_loss::{:06.8f}, Val_loss::{:06.8f}, Trn_F1::{:06.8f}, Val_F1::{:06.8f}"

            .format(j, trn_losses, val_losses, trn_F1, val_F1))
flags = dict()
flags['epochs'] = 15

flags['bs'] = 16

flags['seed'] = 7
xmp.spawn(fit, args=(flags,), nprocs=8, start_method='fork')
def predict(loader, device):

    with torch.no_grad():

        torch.cuda.empty_cache()

        model.eval()

        preds = []

        t = tqdm(loader, leave=False, total=len(loader))

        for i, (ip, _) in enumerate(t):

            ip = ip.to(device)

            output = torch.sigmoid(model(ip))

            preds.append(output.cpu().detach())

        preds = torch.cat(preds)

        return [" ".join(decode_labels(pred)) for pred in preds]
TEST_CSV = '../input/jovian-pytorch-z2g/submission.csv'
test_df = pd.read_csv(TEST_CSV)
test_tf = transforms.Compose([

    transforms.RandomCrop(512, padding=8, padding_mode='symmetric'),

    transforms.ToTensor(),

    normalize

])
test_ds = ProteinDataset(TEST_DIR, test_df, test_tf)
device = xm.xla_device()
model = WRAPPED_MODEL.to(device)
load_checkpoint(model, './best_model.pth')
test_loader = DataLoader(test_ds, 32, num_workers=4, pin_memory=True)
preds = predict(test_loader, device)
preds
len(preds), len(test_df)
sub_df = pd.read_csv(TEST_CSV)
sub_df['Label'] = preds
sub_df.head()
sub_df.to_csv('submission.csv', index=False)