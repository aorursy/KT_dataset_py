# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
import torchvision
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torchvision.transforms as tt
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
%matplotlib inline
from torchvision.utils import save_image
from torchvision.utils import make_grid
from tqdm.notebook import tqdm
data_path = "../input/stanford-cars-dataset/cars_train"
image_size = 256
batch_size = 64
normstats = (0.,0.,0.),(1.,1.,1.)
transforms = tt.Compose([tt.Resize(image_size),#resize to make things uniform
                        tt.CenterCrop(image_size),#cropping to the center to avoid distortion
                        tt.ToTensor(),#to a tensor
                        tt.Normalize(*normstats)#normalizing in order to increase effectiveness of our GAN
                        ])
dataset = ImageFolder(data_path, transform = transforms)
img, _ = dataset[0]
plt.imshow(img.permute((1,2,0)))
def denorm(img_tensors):
    return img_tensors * normstats[1][0] + normstats[0][0]

def show_batch(dl):#just to show one batch of our data
    for img, _ in dl:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xticks([]);ax.set_yticks([])
        ax.imshow(torchvision.utils.make_grid(img[:64],nrow = 8).permute(1,2,0))
        break

def show_images(images):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xticks([]),ax.set_yticks([])
    ax.imshow(torchvision.utils.make_grid(denorm(images.detach()[:64]),nrow = 8).permute(1,2,0))

dataload = DataLoader(dataset,batch_size,num_workers = 4,shuffle = True, pin_memory=True)#makes our data into batches
show_batch(dataload)
def find_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device('cpu')
device = find_default_device()
device
def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    else:
        return data.to(device, non_blocking=True)
class DataloaderDeviced():
    def __init__(self,data,device):
        self.data = data
        self.device = device
    def __iter__(self):
        for b in self.data:
            yield to_device(b,self.device)
    def __len__(self):
        return len(self.data)
dataload = DataloaderDeviced(dataload,device)
descriminator = nn.Sequential(
    #input size being of 3 channels, 256x256
    nn.Conv2d(3, 32 ,kernel_size = 3, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(0.1, inplace=True),
    #output size being of 32 channels, 128x128
    
    nn.Conv2d(32,64,kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.1, inplace=True),
    #out 64x64x64
    
    nn.Conv2d(64,128,kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.1, inplace = True),
    #out 128x32x32
    
    nn.Conv2d(128,256,kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.1,inplace = True),
    #out 256x16x16
    
    nn.Conv2d(256,512, kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.1, inplace = True),
    #out 512x8x8
    
    nn.Conv2d(512,1024, kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(1024),
    nn.LeakyReLU(0.1,inplace = True),
    #out 1024x4x4
    
    nn.Conv2d(1024,1,kernel_size = 4,stride = 1, padding = 0, bias = False),
    #out 1x1x1
    
    nn.Flatten(),
    nn.Sigmoid(),
    #final activation for T/F
)

#descriminator.load_state_dict(torch.load("../input/weights/discweights4.pth"))
descriminator = to_device(descriminator,device)
descriminator
latent_sz = 128
generator = nn.Sequential(
    #latent in 128x1x1
    nn.ConvTranspose2d(128,1024,kernel_size = 4, stride = 1, padding = 0, bias = False),
    nn.BatchNorm2d(1024),
    nn.LeakyReLU(0.1, inplace=True),
    #out 1024x4x4
    
    nn.ConvTranspose2d(1024,512,kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.1, inplace=True),
    #out 512x8x8
    
    nn.ConvTranspose2d(512,256,kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.1, inplace=True),
    #out 256x16x16
    
    nn.ConvTranspose2d(256,128,kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.1, inplace=True),
    #out 128x32x32
    
    nn.ConvTranspose2d(128,64,kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.1, inplace=True),
    #out 64x64x64
    
    nn.ConvTranspose2d(64,32,kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(0.1, inplace=True),
    #out 32x128x128
    
    nn.ConvTranspose2d(32,3,kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.Tanh()
    #out 3x256x256
    
)
#generator.load_state_dict(torch.load("../input/weights/genweights4.pth"))
generator
xb = torch.randn(batch_size,latent_sz,1,1,)
fake_images = generator(xb)
show_images(fake_images)
generator = to_device(generator,device)
def train_discriminator(real_images,opt_d):
    
    opt_d.zero_grad()
    real_preds = descriminator(real_images)
    real_targets = torch.ones(real_images.size(0),1,device = device)
    real_loss = F.binary_cross_entropy(real_preds,real_targets)
    real_score = torch.mean(real_preds).item()
    
    latent = torch.randn(batch_size,latent_sz,1,1, device = device)
    fake_images = generator(latent)
    
    fake_preds = descriminator(fake_images)
    fake_targets = torch.zeros(fake_images.size(0),1,device = device)
    fake_loss = F.binary_cross_entropy(fake_preds,fake_targets)
    fake_score = torch.mean(fake_preds).item()
    
    loss = fake_loss+real_loss
    loss.backward()
    opt_d.step()
    
    return loss.item(),real_score,fake_score
def train_generator(opt_g):
    opt_g.zero_grad()
    latent = torch.randn(batch_size,latent_sz, 1,1, device = device)
    images = generator(latent)
    
    targets = torch.ones(batch_size,1,device = device)
    score = descriminator(images)
    loss = F.binary_cross_entropy(score,targets)
    
    loss.backward()
    opt_g.step()
    
    return loss.item()
savedir = "gen"
os.makedirs(savedir, exist_ok = True)

def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}e1.png'.format(index+90)
    save_image(denorm(fake_images), os.path.join(savedir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
torch.manual_seed(64)
fixed_latent = torch.randn(batch_size, latent_sz, 1, 1, device=device)
save_samples(0,fixed_latent)
def fit(epochs, lr, start_idx = 1):
    loss_d =[]
    loss_g = []
    real_scores = []
    fake_scores = []
    
    opt_d = torch.optim.Adam(descriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for img, _ in tqdm(dataload):
            
            loss, real_score, fake_score = train_discriminator(img, opt_d)
            lossg = train_generator(opt_g)
            
        loss_d.append(loss)
        loss_g.append(lossg)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss, lossg, real_score, fake_score))
        
        save_samples(epoch+start_idx, fixed_latent, show=False)
        
    return loss_g,loss_d,real_scores,fake_scores
lr = 5e-4
epochs = 2
history = fit(epochs,lr)
torch.save(generator.state_dict(),"genweights4.pth")
torch.save(descriminator.state_dict(),"discweights4.pth")
torch.manual_seed(94)
latent_test = torch.randn(1,latent_sz,1,1,device=device)
image = generator(latent_test)
image=to_device(image, torch.device("cpu"))
b=image[0].permute(1,2,0).detach().numpy()
b.shape
plt.imshow(b)
!pip install jovian
prjname="cars_generator"
import jovian
jovian.commit(project=prjname,environment = None)
