import os, sys

import random

import numpy as np 

import pandas as pd



from PIL import Image

from PIL import Image



from datetime import datetime

import torch

import torch.nn as nn

import torch.utils.data as D

import torch.nn.functional as F



import torchvision

from torchvision import transforms as T



import tqdm



import warnings

warnings.filterwarnings('ignore')
! ls -lahtr ../input/recursion-cellular-image-classification/
batch_size = 64
path_data = '../input/recursion-cellular-image-classification/'

trnalldf = pd.read_csv(os.path.join(path_data, 'train.csv'))

tstdf = pd.read_csv(os.path.join(path_data, 'test.csv'))

statsdf = pd.read_csv(os.path.join(path_data, 'pixel_stats.csv'))
trnalldf.iloc[0]
# lets look at the samples per per experiments

tstdf.experiment.value_counts().sort_index()
# Lets take the first 3 experiments as train, and the next 4 as val

valdf = trnalldf[trnalldf.experiment.str.contains('01')]

trndf = trnalldf[trnalldf.experiment.str.contains('02|03|04|05|06|07')]
print('Train frame shape : rows {} cols {}'.format(*trndf.shape))

print('Val frame shape : rows {} cols {}'.format(*valdf.shape))

print('Test frame shape : rows {} cols {}'.format(*tstdf.shape))
statsdf.iloc[0]
meanexpdf = statsdf.groupby(['experiment', 'channel'])['mean'].mean().unstack()

stdexpdf = statsdf.groupby(['experiment', 'channel'])['mean'].mean().unstack()

meanexpdf[meanexpdf.index.str.contains('01|02')]
# We can see large differences by  

meanexpdf.loc['HEPG2-01'].values
from albumentations import (Cutout, Compose, Normalize, RandomRotate90, HorizontalFlip,

                           VerticalFlip, ShiftScaleRotate, Transpose, OneOf, IAAAdditiveGaussianNoise,

                           GaussNoise, RandomGamma, RandomContrast, RandomBrightness, HueSaturationValue,

                           RandomCrop, Lambda, NoOp, CenterCrop, Resize)
def aug(p=1.):

    return Compose([

        RandomRotate90(),

        HorizontalFlip(),

        VerticalFlip(),

        Transpose(),

        NoOp(),

    ], p=p)
class ImagesDS(D.Dataset):

    def __init__(self, df, img_dir, size = 256, mode='train', meandf = meanexpdf, stddf = stdexpdf, channels=[1,2,3,4,5,6]):

        

        self.records = df.to_records(index=False)

        self.channels = channels

        self.site = random.randint(1,2) # load a random site from each well.

        self.mode = mode

        self.meandf = meanexpdf

        self.stddf = stdexpdf

        self.img_dir = img_dir

        self.len = df.shape[0]

        self.size = size

        self.augtransform = aug()

        

    @staticmethod

    def _load_img_as_tensor(file_name, size):

        with Image.open(file_name) as img:

            img = img.resize((size, size), resample=Image.BICUBIC)

            return img

        

    @staticmethod

    def torch_augment(img, transform, mean_, sd_):

        img = img.astype(np.float32)

        img = transform(image = img)['image']

        img = torch.from_numpy(np.moveaxis(img, -1, 0).astype(np.float32))

        img = T.Normalize([*list(mean_)], [*list(sd_)])(img)

        return img  



    def _get_img_path(self, index, channel):

        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate

        return '/'.join([self.img_dir,self.mode,experiment,f'Plate{plate}',f'{well}_s{self.site}_w{channel}.png'])

        

    def __getitem__(self, index):

        paths = [self._get_img_path(index, ch) for ch in self.channels]

        

        # Normalise values

        meanvals = self.meandf.loc[self.records[index].experiment].values

        stdvals = self.stddf.loc[self.records[index].experiment].values

        

        # Load image

        img = np.stack([self._load_img_as_tensor(img_path, self.size) for (img_path, m, s) in zip(paths, meanvals, stdvals)], -1)

        img = self.torch_augment(img, self.augtransform, meanvals, stdvals)      

        

        if self.mode == 'train':

            return img, self.records[index].sirna

        else:

            return img, self.records[index].id_code



    def __len__(self):

        """

        Total number of samples in the dataset

        """

        return self.len
trnloader = D.DataLoader(ImagesDS(trndf, path_data, mode='train'), batch_size=batch_size, shuffle=True, num_workers=4)

valloader = D.DataLoader(ImagesDS(valdf, path_data, mode='train'), batch_size=batch_size*2, shuffle=False, num_workers=4)

tstloader = D.DataLoader(ImagesDS(tstdf, path_data, mode='test'), batch_size=batch_size*2, shuffle=False, num_workers=4)
X,y = next(iter(trnloader))
X.mean(), X.std()
print('Batch Shape : {}'.format(X.shape))

print('Label Shape : {}'.format(y.shape))
class DenseNet(nn.Module):

    def __init__(self, num_classes=1000, num_channels=6):

        super().__init__()

        preloaded = torchvision.models.densenet121(pretrained=True)

        self.features = preloaded.features

        self.features.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3)

        self.classifier = nn.Linear(1024, num_classes, bias=True)

        del preloaded

        

    def forward(self, x, emb=False):

        features = self.features(x)

        out = F.relu(features, inplace=True)

        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)

        if emb:

            return out

        out = self.classifier(out)

        return out
%%capture

model = DenseNet(num_classes=trndf.sirna.max()+1, num_channels = 6)

model
dir(model)[-10:]
# Look at the first 10 layers

[(n, w.shape) for t, (n,w) in enumerate(model.named_parameters()) if t <10]
model.features.denseblock2.denselayer10.conv2.weight[:1]
%%capture

# install NVIDIA Apex if needed to support mixed precision training

use_amp = True

if use_amp:

    try:

        from apex import amp

    except ImportError:

        !pip install  -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/*/*/NVIDIA-apex*

        from apex import amp
device = 'cuda'

model = DenseNet(num_classes=trndf.sirna.max()+1, num_channels = 6)

model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False, loss_scale="dynamic")
# One cycle policy https://sgugger.github.io/the-1cycle-policy.html#the-1cycle-policy

optimizer.param_groups[0]['lr'] = .001 

onecycdf = trnalldf

onecycleloader = D.DataLoader(ImagesDS(onecycdf, path_data, mode='train'), batch_size=batch_size, shuffle=True, num_workers=8)

print('Total Step Count : {}'.format(len(onecycleloader)))

import math

init_value = 1e-5

final_value=1.

beta = 0.98

avg_loss = 0.

batch_num = 0

numsteps = len(onecycleloader)-1

mult = (final_value / init_value) ** (1/numsteps )

lrvals = pd.Series([init_value*(mult**i) for i in  range(numsteps+ 1)])

lrvals.plot(title='LR per step')
lossls = []

optimizer.param_groups[0]['lr'] = init_value

for t, (x, y) in enumerate(onecycleloader): 

    optimizer.zero_grad()

    x = x.to(device)#.half()

    y = y.cuda()

    xgrad = torch.autograd.Variable(x, requires_grad=True)#.half()

    ygrad = torch.autograd.Variable(y)

    out = model(xgrad)

    loss = criterion(out, ygrad)

    with amp.scale_loss(loss, optimizer) as scaled_loss:

        scaled_loss.backward()

    optimizer.step()

    optimizer.param_groups[0]['lr'] = init_value*(mult**t)

    

    ######One Cycle Policy##########>

    #Compute the smoothed loss

    batch_num += 1

    avg_loss = beta * avg_loss + (1-beta) *loss.item()

    smoothed_loss = avg_loss / (1 - beta**batch_num)

    lossls.append(smoothed_loss)



    if t%20==0:

        print('Step {} lr {:.6f} smoothed loss {:.5f} time {}'.format(t, init_value*(mult**t), smoothed_loss, datetime.now()))

    del loss, out, y, x# , target
pd.Series(lossls, index=np.log10(lrvals)).plot(title='Smoothed loss per LR (log10)', ylim=(6.9,7.3), figsize = (10,4))
%%capture

!pip install https://github.com/ceshine/pytorch_helper_bot/archive/0.0.5.zip

from helperbot import GradualWarmupScheduler

from torch.optim.lr_scheduler import _LRScheduler

from torch.optim.lr_scheduler import ReduceLROnPlateau
EPOCHS=8

lrmult=10

lr = 3e-2/lrmult

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=lrmult, total_epoch=2, after_scheduler=scheduler_cosine)
lrls = []

for e in range(EPOCHS):

    scheduler_warmup.step()

    lrls.append(scheduler_warmup.get_lr()[0])

pd.Series(lrls).plot(title='Learning Rate by epoch')
model = DenseNet(num_classes=trndf.sirna.max()+1, num_channels = 6)

model.to(device)

model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False, loss_scale="dynamic")

scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=lrmult, total_epoch=2, after_scheduler=scheduler_cosine)
@torch.no_grad()

def prediction(model, loader):

    probs = []

    for x, _ in tqdm.tqdm(loader):

        x = x.to(device)

        output = model(x)

        outmat = torch.sigmoid(output.cpu()).numpy()

        probs.append(outmat)

    probs = np.concatenate(probs, 0)

    return probs
for epoch in range(EPOCHS):

    tloss = 0.

    model.train()

    scheduler_warmup.step()

    for t, (x, y) in tqdm.tqdm(enumerate(trnloader)): 

        optimizer.zero_grad()

        x = x.to(device)#.half()

        y = y.cuda()

        xgrad = torch.autograd.Variable(x, requires_grad=True)#.half()

        ygrad = torch.autograd.Variable(y)

        out = model(xgrad)

        loss = criterion(out, ygrad) 

        with amp.scale_loss(loss, optimizer) as scaled_loss:

            scaled_loss.backward()

        optimizer.step()

        tloss += loss.item() 

        del loss, out, y, x

    print('Epoch {} -> Train Loss: {:.4f} -> LR: {:.5f} -> Time {}'.format(epoch+1, tloss/len(trnloader), scheduler_warmup.get_lr()[0], datetime.now()))

    

    model.eval()

    preds = prediction(model, valloader)

    val_accuracy = (valdf.sirna.values == preds.argmax(1)).mean()

    print('Epoch {} -> Val Acc: {:.4f} -> Time {}'.format(epoch+1, val_accuracy, datetime.now()))

output_model_file = "recursion_model.bin"

torch.save(model.state_dict(), output_model_file)
## Load up the model

# model.load_state_dict(torch.load(os.path.join(path_data, "recursion_model.bin")))

# model.to(device)

# for param in model.parameters():

#     param.requires_grad = False

# model.eval()

# ....