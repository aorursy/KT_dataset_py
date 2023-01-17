import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import os
import glob
import PIL
from PIL import Image
import itertools
import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid

cuda = torch.cuda.is_available()

if cuda:
    print("Cuda ON")
else:
    print("No GPU...")
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDataset(Dataset):
    def __init__(self, root, transform_, aligned, mode='train'):
        self.transform = transforms.Compose(transform_)
        self.aligned = aligned
        self.files_A = sorted(glob.glob(os.path.join(root, "{}A".format(mode)) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "{}B".format(mode)) + "/*.*"))
        
    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        
        if self.aligned:
            image_B = Image.open(self.files_B[index % len(self.files_B)])
        else:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B)-1)])
            
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)
            
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        
        return {"A": item_A, "B": item_B}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
class residual_block(nn.Module):
    def __init__(self, in_features):
        super(residual_block, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
        
    def forward(self, x):
        return x + self.block(x)
    
class GeneratorResnet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResnet, self).__init__()
        
        channels = input_shape[0]
        out_features = 64
        
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        
        # DownSample
        for _ in range(2):
            in_features = out_features
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            
        # Residual Blocks for 128 x 128 // for 256x256 or higher, use 9 such blocks
        for _ in range(num_residual_blocks):
            model += [residual_block(out_features)]
            
        # Upsample
        for _ in range(2):
            in_features = out_features
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            
        # Output Layer
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features, channels, 7),
            nn.InstanceNorm2d(channels),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        out_features, (channels, height, width) = 64, input_shape
        self.output_shape = (1, height // 2**4, width // 2**4)
        model = [
            nn.Conv2d(channels, out_features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        for _ in range(3):
            in_features = out_features
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            
        model += [nn.ZeroPad2d((1,0,1,0))]
        model += [nn.Conv2d(out_features, 1, 4, padding=1)]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
    
    def step(self, epoch):
        return 1.0 - max(0, epoch+self.offset-self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty Buffer Not Allowed"
        self.max_size = max_size
        self.data = []
        
    def push_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))
# Compile Loss Functions
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

img = np.asarray(Image.open('/kaggle/input/monet2photo/monet2photo/trainA/0.jpg')) # Hardcoded for Kaggle Kernel
input_shape = (img.shape[2], img.shape[0], img.shape[1])

G_AB = GeneratorResnet(input_shape, num_residual_blocks=9)
G_BA = GeneratorResnet(input_shape, num_residual_blocks=9)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    G_AB, G_BA, D_A, D_B = G_AB.cuda(), G_BA.cuda(), D_A.cuda(), D_B.cuda()
    criterion_GAN, criterion_cycle, criterion_identity = criterion_GAN.cuda(), criterion_cycle.cuda(), criterion_identity.cuda()
    
G_AB.apply(weight_init)
G_BA.apply(weight_init)
D_A.apply(weight_init)
D_B.apply(weight_init)

# ------------ #
#  Hyperparams
# ------------ #

lr = 0.0002
beta1, beta2 = 0.5, 0.999
n_epochs = 200
epoch = 0
decay_epoch = 100

optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(beta1, beta2))
optimizer_DA = optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_DB = optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, beta2))

lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_DA = optim.lr_scheduler.LambdaLR(optimizer_DA, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_DB = optim.lr_scheduler.LambdaLR(optimizer_DB, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

fakeA_buffer = ReplayBuffer()
fakeB_buffer = ReplayBuffer()

transform_ = [
    transforms.Resize(int(input_shape[1]*1.12), Image.BICUBIC),
    transforms.RandomCrop((input_shape[1], input_shape[2])),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
]

trainloader = DataLoader(
    ImageDataset("/kaggle/input/monet2photo/monet2photo/", transform_=transform_, aligned=False, mode='train'),
    batch_size = 1,
    shuffle=True
)

# validloader = DataLoader(
#     ImageDataset("/kaggle/input/monet2photo/monet2photo/", transform_=transform_, aligned=False, mode='test'),
#     batch_size=1,
#     shuffle=True
# )
def sample_image(finished_batch):
    imgs = next(iter(validloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs['A'].type(Tensor))
    real_B = Variable(imgs['B'].type(Tensor))
    fake_A = G_BA(real_B)
    fake_B = G_AB(real_A)
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s.png" % finished_batch, normalize=False)

from collections import defaultdict
t0 = time.time()
output = defaultdict(list)

for epoch in range(n_epochs):
    for i, batch in enumerate(trainloader):
        
        real_A = Variable(batch['A'].type(Tensor))
        real_B = Variable(batch['B'].type(Tensor))
        
        t = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
        f = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)
        
        # --------------- #
        # Train Generator
        # --------------- #
        
        optimizer_G.zero_grad()
        G_AB.train()
        G_BA.train()
        
        # Identity Loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)
        
        loss_identity = (loss_id_A + loss_id_B) / 2
        
        fake_A, fake_B = G_BA(real_B), G_AB(real_A)
        
        # GAN Loss
        loss_GAN_AB = criterion_GAN(D_B(fake_B), t)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), t)
        
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
        
        # Cycle-consistency Loss
        loss_cycle_A = criterion_cycle(G_BA(fake_B), real_A)
        loss_cycle_B = criterion_cycle(G_AB(fake_A), real_B)
        
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
        
        # Hyperparameters in the literature
        loss_G = 5*loss_identity + 10*loss_cycle + loss_GAN
        
        loss_G.backward()
        optimizer_G.step()
        
        # ------------------- #
        # Train Discriminator
        # ------------------- #
        
        G_AB.eval()
        G_BA.eval()
        
        optimizer_DA.zero_grad()
        
        loss_real_DA = criterion_GAN(D_A(real_A), t)
        fake_A = fakeA_buffer.push_pop(fake_A)
        loss_fake_DA = criterion_GAN(D_A(fake_A), f)
        loss_DA = (loss_real_DA + loss_fake_DA) / 2
        loss_DA.backward()
        optimizer_DA.step()
        
        optimizer_DB.zero_grad()
        
        loss_real_DB = criterion_GAN(D_B(real_B), t)
        fake_B = fakeB_buffer.push_pop(fake_B)
        loss_fake_DB = criterion_GAN(D_B(fake_B), f)
        loss_DB = (loss_real_DB + loss_fake_DB) / 2
        loss_DB.backward()
        optimizer_DB.step()
        
        loss_D = (loss_DA + loss_DB) / 2
        
    t1 = time.time()
    
    output['epoch'].append(epoch+1)
    output['Loss_G'].append(loss_G.item())
    output['Loss_D'].append(loss_D.item())
    output['Loss_id'].append(loss_identity.item())
    output['Loss_GAN'].append(loss_GAN.item())
    output['Loss_cycle'].append(loss_cycle.item())
    
    print("Epoch: {}".format(epoch+1))
    print("Time per Epoch: {:.1f}m".format((t1-t0)/60))
    print("Generator Loss: {:.3f}".format(loss_G.item()))
    print("Discriminator Loss: {:.3f}".format(loss_D.item()))
    print("Identity Loss: {:.3f}".format(loss_identity.item()))
    print("GAN Loss: {:.3f}".format(loss_GAN.item()))
    print("Cycle Loss: {:.3f}".format(loss_cycle.item()))
    print('-'*50)
