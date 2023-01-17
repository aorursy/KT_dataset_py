!pip install pytorch_lightning
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import sampler
from torch.optim import SGD, Adam

import torchvision.models as models

from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning import Trainer

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

import PIL
from PIL import Image
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
import os
image_name_list = []
for dirname, _, filenames in os.walk('/kaggle/input/animed/cropped'):
    for filename in filenames:
        image_name_list.append(os.path.join(dirname, filename))
size = 400
w = 80

images = []
i = 0
j = 0
with tqdm(total=size, unit='img') as pbar:
    while i < size:
        j += 1
        try:
            image = Image.open(image_name_list[j])
        except IOError:
            continue
            
        image = image.resize((w, w))
        image = np.array(image)
        image = image / 255
        image = image.transpose([2, 0, 1])
        images.append(image)
        i += 1
        pbar.update(1)
        
train_dataset = TensorDataset(torch.tensor(np.array(images)))
class Flatten(nn.Module):
    
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)  
    
class Unflatten(nn.Module):
    
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)
epoch         = 1000
batch_size    = 100
noise_size    = 296
lr_gen        = 1e-3
lr_dis        = 1e-3
verbose       = 100
dtype         = torch.cuda.FloatTensor

class AnimeModel(pl.LightningModule):

    def __init__(self):
        
        super(AnimeModel, self).__init__()
        
        self.D = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size = 5, stride = 1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64,kernel_size = 5, stride = 1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2,2),
            Flatten(),
            nn.Linear(18496, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024,1)
            
        ).type(dtype)
        
        self.G = nn.Sequential(
            
            nn.Linear(noise_size,1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 8*w*w),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(8*w*w),
            Unflatten(batch_size, 128, w // 4, w // 4),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            
        ).type(dtype)
        
        self.iter = 0

    def training_step(self, batch, batch_nb, optimizer_idx):
        
        self.iter += 1
        
        if self.iter % verbose == 0:
            
            real_images, = batch
            
            g_fake_seed     = self.sample_noise(batch_size, noise_size).type(dtype)
            fake_images     = self.G(g_fake_seed)
            
            fig = plt.figure()
            plt.imshow(fake_images[0].permute(1, 2, 0).cpu().detach())
            plt.show()
            
            self.logger.experiment.add_image("fake_image0", fake_images[0])
            self.logger.experiment.add_image("fake_image1", fake_images[1])
            self.logger.experiment.add_image("fake_image2", fake_images[2])
            self.logger.experiment.add_image("fake_image3", fake_images[3])
            
            self.logger.experiment.add_image("real_image0", real_images[0])
            self.logger.experiment.add_image("real_image1", real_images[1])
            self.logger.experiment.add_image("real_image2", real_images[2])
            self.logger.experiment.add_image("real_image3", real_images[3])
        
        if optimizer_idx == 0:
            
            g_fake_seed     = self.sample_noise(batch_size, noise_size)
            fake_images     = self.G(g_fake_seed)

            gen_logits_fake = self.D(fake_images)
            g_error         = self.generator_loss(gen_logits_fake)
            
            return {
                "loss"         : g_error,
                'progress_bar' : {'gen_loss': g_error},
                'log'          : {'gen_loss': g_error}
            }
            
        
        if optimizer_idx == 1:
            
            X,              = batch
            real_data       = X.type(dtype)
            logits_real     = self.D(2* (real_data - 0.5))
        
            g_fake_seed     = self.sample_noise(batch_size, noise_size)
            fake_images     = self.G(g_fake_seed).detach()
            logits_fake     = self.D(fake_images)

            d_total_error   = self.discriminator_loss(logits_real, logits_fake)
            
            return {
                "loss"         : d_total_error,
                'progress_bar' : {'disc_loss': d_total_error},
                'log'          : {'disc_loss': d_total_error}
            }
        
    def forward(self):
        g_fake_seed = self.sample_noise(1, noise_size)
        return self.G(g_fake_seed)
   
    def configure_optimizers(self):
        generator_opt    = Adam(self.G.parameters(), lr = lr_gen,  betas=(0.5, 0.999))
        disriminator_opt = Adam(self.D.parameters(), lr = lr_dis,  betas=(0.5, 0.999))
        return generator_opt, disriminator_opt

    def sample_noise(self, batch_size, dim):
        return (torch.rand([batch_size, dim])*2 - 1).type(dtype)
    
    def bce_loss(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()
    
    def discriminator_loss(self, logits_real, logits_fake):
        N = logits_real.shape[0]
        
        true_labels  = torch.ones(N).type(dtype)
        false_labels = torch.zeros(N).type(dtype)
        
        loss   = self.bce_loss(logits_real, true_labels) + self.bce_loss(logits_fake, false_labels)
        
        return loss
    
    def generator_loss(self, logits_fake):
        N = logits_fake.shape[0]
        
        true_labels = torch.ones(N).type(dtype)
        
        loss = self.bce_loss(logits_fake, true_labels)
        return loss
    
    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=batch_size)
model = AnimeModel()
trainer = Trainer(early_stop_callback=False, max_nb_epochs=epoch)
trainer.fit(model)
g_fake_seed = model.sample_noise(batch_size, noise_size)
fake_images = model.G(g_fake_seed)
plt.imshow(fake_images[12].permute(1, 2, 0).cpu().detach())
torch.save(model.G, "AnimeGenerator.pth")