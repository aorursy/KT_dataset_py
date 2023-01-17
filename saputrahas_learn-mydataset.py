%matplotlib inline

import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

import PIL

import PIL.Image

import numpy as np

import matplotlib.pyplot as plt

import os

import glob

import cv2

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
!ls -all /kaggle/input/dataset/dataset/
path = Path('/kaggle/input/dataset/dataset/')

train_path = path.joinpath('train')

dirfiles = sorted(list(train_path.glob('*')))

files = sorted(list(train_path.glob('*/*.jpg')))
trainset = torchvision.datasets.ImageFolder(train_path)

print('Lihat Gambar Train :', trainset.__len__())

image, label = trainset.__getitem__(0)

trainset.class_to_idx
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, path, transform_input=None, transform_target=None):

        self.path = Path(path)

        self.dirfiles = sorted(list(self.path.glob('*')))

        self.files = sorted(list(self.path.glob('*/*.jpg')))

        self.transform_input = transform_input

        self.transform_target = transform_target

    

    def __len__(self):

        return len(self.files)

    

    def __getitem__(self, idx):

        pf = str(self.files[idx])

        dirname = pf.split('/')[-2]

        img_input = PIL.Image.open(pf).convert("RGB")

        img_target = PIL.Image.open(pf).convert("RGB")

        if self.transform_input:

            img_input = self.transform_input(img_input)

        if self.transform_target:

            img_target = self.transform_target(img_target)

        return img_input,img_target





tp = '/kaggle/input/dataset/dataset/train'

train_dataset = MyDataset(tp)



vp = '/kaggle/input/dataset/dataset/valid'

valid_dataset = MyDataset(vp)
BATCH_SIZE = 16

NUM_EPOCH = 20
tminput = transforms.Compose([

    transforms.Resize((90,128)),

    transforms.Resize((352,512)),

    transforms.ToTensor(),

])



tmtarget = transforms.Compose([

    transforms.Resize((352,512)),

    transforms.ToTensor(),

])



trp = '/kaggle/input/dataset/dataset/train'

vlp = '/kaggle/input/dataset/dataset/valid'



trainset = MyDataset(trp, transform_input=tminput, transform_target=tmtarget)

validset = MyDataset(vlp, transform_input=tminput, transform_target=tmtarget)



train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

valid_loader = torch.utils.data.DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
class StdUpsample(nn.Module):

    def __init__(self, nin, nout):

        super().__init__()

        self.upfac = 2

        self.pixel_shuffle = nn.PixelShuffle(self.upfac)

        self.conv = nn.Conv2d(nin, nout*self.upfac**2, kernel_size=3, stride=1, padding=1)

        self.bn = nn.InstanceNorm2d(nout, affine=True)

        

    def forward(self, x):

        x = self.conv(x)

        x = self.pixel_shuffle(x)

        x = F.leaky_relu(x)

        x = self.bn(x)

        return x

    

class ColorNet(nn.Module):

    def __init__(self, resnet):

        super(ColorNet, self).__init__()

        self.resnet = resnet

        

        #encoder

        self.input_layer = nn.Sequential(

            resnet.conv1,

            resnet.bn1,

            resnet.relu,

        )

        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1

        self.layer2 = resnet.layer2

        self.layer3 = resnet.layer3

        self.layer4 = resnet.layer4

        

        num_feat = self.layer4[-1].bn2.num_features

        

        self.up1 = StdUpsample(num_feat, num_feat//2)

        self.up2 = StdUpsample((num_feat//2+resnet.layer3[-1].bn2.num_features), num_feat//4)

        self.up3 = StdUpsample((num_feat//4+resnet.layer2[-1].bn2.num_features), num_feat//8)

        self.up4 = StdUpsample((num_feat//8+resnet.layer1[-1].bn2.num_features), num_feat//16)

        self.up5 = StdUpsample((num_feat//16 + resnet.conv1.out_channels), num_feat//32)

        

        self.output_layer = nn.Conv2d(num_feat//32, 3, kernel_size=1)  

    

    def freeze_encoder(self):

        for param in self.input_layer.parameters():

            param.requires_grad = False

        for param in self.layer1:

            param.requires_grad = False

        for param in self.layer2:

            param.requires_grad = False

        for param in self.layer3:

            param.requires_grad = False

        for param in self.layer4:

            param.requires_grad = False

            

    def unfreeze_encoder(self):

        for param in self.input_layer.parameters():

            param.requires_grad = True

        for param in self.layer1:

            param.requires_grad = True

        for param in self.layer2:

            param.requires_grad = True

        for param in self.layer3:

            param.requires_grad = True

        for param in self.layer4:

            param.requires_grad = True

    

    def forward(self, x):

        # encoder

        el1 = self.input_layer(x)

        mp1 = self.maxpool(el1)

        

        el2 = self.layer1(mp1)

        

        el3 = self.layer2(el2)

        el4 = self.layer3(el3)

        el5 = self.layer4(el4)

        

        #decoder

        dl1 = self.up1(el5)

        cat1 = torch.cat([el4, dl1], dim=1)

        dl2 = self.up2(cat1)

        

        cat2 = torch.cat([el3, dl2], dim=1)

        dl3 = self.up3(cat2)

        

        cat3 = torch.cat([el2, dl3], dim=1)

        dl4 = self.up4(cat3)

        

        cat4 = torch.cat([el1, dl4], dim=1)

        dl5 = self.up5(cat4)

        

        out = self.output_layer(dl5)

        return out

            

resnet = torchvision.models.resnet34(pretrained=True)

model = ColorNet(resnet)

model = model.cuda()

model.freeze_encoder()
train_iter = iter(train_loader)

img_input, img_target = train_iter.next()

output = model(img_input.cuda())
idx = 2

img = output[idx].permute(1,2,0).cpu().detach().numpy()

plt.imshow(img_target[idx].permute(1,2,0));plt.show();

plt.imshow(img_input[idx].permute(1,2,0));plt.show();

plt.imshow(img);plt.show();
class AverageMeter(object):

    """Computes and stores the average and current value"""

    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count
import torch.optim as optim

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()

for epoch in range(NUM_EPOCH):

    train_loss = AverageMeter()

    valid_loss = AverageMeter()

    trtdata = train_loader.dataset.__len__()

    vltdata = valid_loader.dataset.__len__()

    

    for idx, (img_input, img_target) in enumerate(train_loader):

        tbatch = trtdata//img_input.size(0)

        img_input = img_input.cuda()

        img_target = img_target.cuda()

        optimizer.zero_grad()

        output = model(img_input)

        trloss = criterion(output, img_target)

        trloss.backward()

        optimizer.step()

        train_loss.update(trloss.data)

        if idx == len(train_loader)-1: 

            print(f'Epoch {epoch+1} \tBatch ({(idx+1)})/({tbatch}) \tTrain Loss: {trloss.data:.6f} ({train_loss.avg:.6f})')

        

    with torch.no_grad():

        for idx, (img_input, img_target) in enumerate(valid_loader):

            tbatch = vltdata//img_input.size(0)

            img_input = img_input.cuda()

            img_target = img_target.cuda()

            output = model(img_input)

            vlloss = criterion(output, img_target)

            valid_loss.update(vlloss.data)

            if idx == len(valid_loader)-1: 

                print(f'Epoch {epoch+1} ({(idx+1)})/({tbatch}) \tBatch  \tTrain Loss: {trloss.data:.6f} ({train_loss.avg:.6f}) \tValid Loss: {vlloss.data:.6f} ({valid_loss.avg:.6f})')
# %load_ext tensorboard.notebook

# %tensorboard --logdir logs

# from torch.utils.tensorboard import SummaryWriter

# import numpy as np



# writer = SummaryWriter ()



# for n_iter in range ( 15 ):

#     writer . add_scalar ( 'Loss/train' , np . random . random (), n_iter )

#     writer . add_scalar ( 'Loss/test' , np . random . random (), n_iter )

#     writer . add_scalar ( 'Accuracy/train' , np . random . random (), n_iter )

#     writer . add_scalar ( 'Accuracy/test' , np . random . random (), n_iter )