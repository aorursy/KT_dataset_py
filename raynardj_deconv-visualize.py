# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import sys
from pathlib import Path
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
from torch import nn
from types import MethodType

from glob import glob

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as trf
from PIL import Image
DATA = Path("/kaggle/input/imagenetmini-1000/imagenet-mini/")
TRAIN = DATA/"train"
VAL = DATA/"val"
imgs=glob(str(TRAIN/"*"/"*"))
len(imgs)
!pip install forgebox
trans = trf.Compose([
    trf.Resize((224,224)),
    trf.ToTensor(),
    trf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
class image_list(Dataset):
    def __init__(self, l,transform = trans):
        self.l = l
        self.transform = transform
        
    def __len__(self):
        return len(self.l)
    
    def __getitem__(self,idx):
        img = Image.open(self.l[idx],).convert('RGB')
        return self.transform(img)
from forgebox.ftorch.prepro import test_DS
test_x = test_DS(image_list(imgs))()
test_x.shape
from torchvision.models import vgg19_bn
fwd_md = vgg19_bn(pretrained = True)
for l in fwd_md.features:
    if l.__class__ == nn.MaxPool2d:
        l.return_indices = True
def new_forward(self,x):
    outputs = []
    indices = []
    ct = 0
    for l in self.features:
        if l.__class__ == nn.MaxPool2d:
#             print(f"{ct} shape before:{x.shape}",end="\t")
            x,ind = l(x)
            outputs.append(x)
            indices.append(ind)
#             print(f"{l.__class__}:\tshape after:{x.shape}\n\n")
        else:
#             print(f"{ct} shape before:{x.shape}",end="\t")
            x = l(x)
#             print(f"{l.__class__}:\tshape after:{x.shape}")
        ct+=1
    return outputs,indices
setattr(fwd_md,"forward",MethodType(new_forward,fwd_md))
y_s,mp_id = fwd_md(test_x)
list(map(lambda x:x.shape,y_s))

for mpi in mp_id:
    print(mpi.shape)
class DeconvBlock(nn.Module):
    def __init__(self,in_,out_,conv_nb = 2,is_unpool = True,final_relu = True):
        super().__init__()
        self.in_ = in_
        self.out_ = out_
        self.is_unpool = is_unpool
        
        conv_layers = list([nn.Conv2d(in_,in_,kernel_size = 3,padding=3//2),
            nn.BatchNorm2d(in_),
            nn.ReLU(),
            nn.Conv2d(in_,out_ if conv_nb==2 else in_,kernel_size = 3,padding=3//2),
            nn.BatchNorm2d(out_ if conv_nb==2 else in_),])
        
        if conv_nb ==3:
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Conv2d(in_,out_,kernel_size = 3,padding=3//2))
            conv_layers.append(nn.BatchNorm2d(out_))
        
        if final_relu:
            conv_layers.append(nn.ReLU())
    
        self.unpool = nn.MaxUnpool2d(2)
        self.convs = nn.Sequential(*conv_layers)
        
    def forward(self,x,max_indices):
        if self.is_unpool:
            x = self.unpool(x,indices = max_indices)
        return self.convs(x)
class DeconvVgg19(nn.Module):
    def __init__(self,):
        super().__init__()
        self.decon_b1 = DeconvBlock(64,3,conv_nb = 2,is_unpool = True,final_relu = False)
        self.decon_b2 = DeconvBlock(128,64,conv_nb = 2)
        self.decon_b3 = DeconvBlock(256,128,conv_nb = 3)
        self.decon_b4 = DeconvBlock(512,256,conv_nb = 3)
        self.decon_b5 = DeconvBlock(512,512,conv_nb = 3)
        self.layers = [
            self.decon_b1, self.decon_b2, self.decon_b3,self.decon_b4,self.decon_b5
        ]
    def forward(self,x,max_indices):
        for i in range(5):
            x = self.layers[-(i+1)](x,max_indices[-(i+1)])
        return x
decon = DeconvVgg19()
crit = nn.MSELoss()
from forgebox.ftorch.train import Trainer
from forgebox.ftorch.callbacks import stat
train_ds=image_list(glob(str(TRAIN/"*"/"*")))
valid_ds=image_list(glob(str(VAL/"*"/"*")))
opt = torch.optim.AdamW(decon.parameters())
t = Trainer(train_ds,
            val_dataset=valid_ds,
            shuffle = True, 
            batch_size = 32,
            opts = [opt],
            callbacks=[stat],
            val_callbacks=[stat])
fwd_md = fwd_md.eval()
if torch.cuda.is_available():
    decon = decon.cuda()
    fwd_md = fwd_md.cuda()
@t.step_train
def action(batch):
    batch.opt.zero_all()
    x = torch.stack(batch.data)
    if torch.cuda.is_available():
        x = x.cuda()
    with torch.no_grad():
        act_s,max_s = fwd_md(x)
    states = [x,]+list(map(lambda x:x.detach(),act_s))
    
    # optimize by block ================
    losses = []
    for i in range(5):
        maxi = max_s[i]
        t_in = states[i+1]
        t_out = states[i]
        t_out_ = decon.layers[i](t_in, maxi.detach())
        losses.append(crit(t_out_,t_out))
    loss = torch.stack(losses).sum()
        
    loss.backward()
    batch.opt.step_all()
    
    # optimize by entire net ================
    batch.opt.zero_all()
    x_ = decon(states[-1].detach(),max_s)
    
    # loss of image reconstruction
    li = crit(x_,x)
    li.backward()
    batch.opt.step_all()
    rt = dict((f"L{i}",losses[i].item()) for i in range(5))
    rt.update(dict(LI = li.item()))
    return rt

@t.step_val
def val_action(batch):
    if batch.i ==0:
        torch.save(decon.state_dict(),f"deconv_e{batch.epoch}_b{batch.i}.pth")
    x = torch.stack(batch.data)
    if torch.cuda.is_available():
        x = x.cuda()
    act_s,max_s = fwd_md(x)
    states = [x,]+list(map(lambda x:x.detach(),act_s))
    
    losses = []
    for i in range(5):
        maxi = max_s[i]
        t_in = states[i+1]
        t_out = states[i]
        t_out_ = decon.layers[i](t_in, maxi.detach())
        losses.append(crit(t_out_,t_out))
    # calc by entire net
    x_ = decon(states[-1].detach(),max_s)
    li = crit(x_,x)
    rt = dict((f"L{i}",losses[i].item()) for i in range(5))
    rt.update(dict(LI = li.item()))
    return rt
t.train(2)
decon = decon.cpu()
fwd_md = fwd_md.cpu()
class DeNorm:
    def __init__(self, m,s):
        self.m = torch.Tensor(m)[:,None,None]
        self.s = torch.Tensor(s)[:,None,None]
        
    def __call__(self,x):
        return torch.clamp((x*self.s)+self.m,0,1)
toimg = trf.Compose([
    DeNorm(m=[0.485, 0.456, 0.406], s=[0.229, 0.224, 0.225]),
    trf.ToPILImage(),
])
gen = iter(DataLoader(train_ds,batch_size = 1,shuffle=True))
next(gen).shape
x = next(gen)
img = toimg(x[0])
display(img)
with torch.no_grad():
    act_s,max_s = fwd_md(x)
    
act = act_s[-1]
hot = (act>(act.max()*0.6)).float()
print(hot.sum())
toimg(decon(hot,max_s)[0])
