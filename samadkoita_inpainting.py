from __future__ import print_function
%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec




manualSeed = 999

print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

imagefolder = '/kaggle/input/celeba-dataset/img_align_celeba'
print(os.listdir(imagefolder))

out_folder = "/kaggle/working/models/celeba/"

batch_size = 128
image_size = 64
nc = 3
noise_dim = 100
num_epochs = 0
lr = 0.0002
beta1 = 0.5
iter_check = 2000
print_check = 100
demo_batch_size = 5
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)

NUM_TRAIN = 200000
dataset = dset.ImageFolder(root=imagefolder,transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

print(len(dataset))
val_mask  = [i for i in range(NUM_TRAIN,len(dataset))]
print(val_mask[-1])
val_set = torch.utils.data.Subset(dataset,val_mask)


from models import Generator,Discriminator

pretrained = torch.load("../input/pretrained/celeba_epoch31.pt")
Gen_eval = Generator().to(device)
Dis_eval = Discriminator().to(device) 

Gen_eval.load_state_dict(pretrained["state_dict_G"])
Dis_eval.load_state_dict(pretrained["state_dict_D"])

Gen_eval.eval()
Dis_eval.eval()

for p in Gen_eval.parameters():
    p.requires_grad=False
    
for p in Dis_eval.parameters():
    p.requires_grad=False
    
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import torch

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

class ImageMask(torch.utils.data.Dataset):
  def __init__(self, dataset, image_size=(3,64,64),param=0.4):
    self.dataset = dataset
    self.image_size = image_size
    self.image_shape = (image_size[1],image_size[2])
    self.map = ['left','random','center','box','up','down']
    self.param = param
  
  def __getitem__(self, index):
    target_image = self.dataset[index][0]
    assert(torch.is_tensor(target_image))
    
    maskType = self.map[index%4]
    mask = np.ones(self.image_shape)
    if index%8==0:
        maskType = 'right'
    

    param = 0.6
    if maskType == 'random':
        mask[np.random.random(self.image_shape) < param] = 0.0
    elif maskType == 'center':
        centerScale = 0.3
        sz = tuple([(int)(z * centerScale) for z in self.image_shape])
        mask[ sz[1]:-sz[1], sz[0]:-sz[0]] = 0.0
    elif maskType == 'left':
        sz = np.random.randint(10,64-35,size=(2,))
        mask[sz[0]:sz[0]+10,sz[1]:sz[1]+30] = 0.0
    elif maskType == 'right':
        sz = np.random.randint(10,64-35,size=(2,))
        mask[sz[1]:sz[1]+30,sz[0]:sz[0]+10] = 0.0
    elif maskType == 'box':
        sz = np.random.randint(10,64-20,size=(3,2))
        mask[sz[0][0]:sz[0][0]+10,sz[0][1]:sz[0][1]+10] = 0.0
        mask[sz[1][0]:sz[1][0]+10,sz[1][1]:sz[1][1]+10] = 0.0
        mask[sz[2][0]:sz[2][0]+10,sz[2][1]:sz[2][1]+10] = 0.0
    elif maskType == 'up':
        c = self.image_shape[0] // 2
        mask[:,:c] = 0.0
    elif maskType == 'down':
        c = self.image_shape[0] // 2
        mask[:,c:] = 0.0
    else:
        assert(False)

    return (target_image, torch.FloatTensor(mask),maskType)
  
  def __len__(self):
    return len(self.dataset)
masked_dataset = ImageMask(val_set)
masked_loader = torch.utils.data.DataLoader(masked_dataset, batch_size=12,
                                         shuffle=False,drop_last=True ,num_workers=2,pin_memory=True)
dataset_fun = dset.ImageFolder(root="../input/images-fun",transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
masked_dataset_fun = ImageMask(dataset_fun)
masked_loader = torch.utils.data.DataLoader(masked_dataset_fun, batch_size=12,
                                         shuffle=False,drop_last=True ,num_workers=2,pin_memory=True)
WINDOW_SIZE = 7
prior_loss_parameter = 0.003
special_conv = torch.ones(1,1,WINDOW_SIZE,WINDOW_SIZE,device=device)
# https://github.com/moodoki/semantic_image_inpainting/blob/extensions/src/inpaint.py
#
#
# Original code from https://github.com/parosky/poissonblending 
import scipy.sparse
import os
import PIL.Image
import torch
from torchvision.transforms import ToPILImage,ToTensor
try:
    import pyamg
except:
    os.system("pip3 install pyamg")
import pyamg


def convert_to_np(pic):
    npimg = None
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if isinstance(pic, torch.Tensor):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))
        
    assert npimg is not None
    
    return npimg


# pre-process the mask array so that uint64 types from opencv.imread can be adapted
def unnormalise(img):
    return img/2 + 0.5

def poissonblending_batch(og_batch,gan_batch,mask_inv_batch):
    out = torch.zeros_like(gan_batch)
    i = 0
    for og,gan,mask_inv in zip(og_batch,gan_batch,mask_inv_batch):
        out[i,:,:,:] = ToTensor()(poissonblending_fromcpu(og,gan,mask_inv))
        i += 1
        
    return out
        

def poissonblending_fromcpu(imgt, imgs, mask):
    mask = convert_to_np(mask)
    mask = np.squeeze(mask)
    imgt = convert_to_np(imgt)
    imgs = convert_to_np(imgs)
    
    assert imgs.shape==(64,64,3)
    assert mask.shape==(64,64)
    
    assert np.min(mask)==0 
    assert np.max(mask)==255
    
    return blend(imgt, imgs,mask)

def blend(img_target, img_source, img_mask, offset=(0, 0)):
    # compute regions to be blended
    region_source = (0,0,64,64)
    region_target = (0,0,64,64)
    region_size = (region_source[2]-region_source[0], region_source[3]-region_source[1])

    # clip and normalize mask image
    img_mask[img_mask==0] = False
    img_mask[img_mask!=False] = True


    # create coefficient matrix
    A = scipy.sparse.identity(np.prod(region_size), format='lil')
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if img_mask[y,x]:
                index = x+y*region_size[1]
                A[index, index] = 4
                if index+1 < np.prod(region_size):
                    A[index, index+1] = -1
                if index-1 >= 0:
                    A[index, index-1] = -1
                if index+region_size[1] < np.prod(region_size):
                    A[index, index+region_size[1]] = -1
                if index-region_size[1] >= 0:
                    A[index, index-region_size[1]] = -1
    A = A.tocsr()
    
    # create poisson matrix for b
    P = pyamg.gallery.poisson(img_mask.shape)

    # for each layer (ex. RGB)
    for num_layer in range(img_target.shape[2]):
        # get subimages
        t = img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer]
        s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3],num_layer]
        t = t.flatten()
        s = s.flatten()

        # create b
        b = P * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask[y,x]:
                    index = x+y*region_size[1]
                    b[index] = t[index]

        # solve Ax = b
        x = pyamg.solve(A,b,verb=False,tol=1e-10)

        # assign x to target image
        x = np.reshape(x, region_size)
        x[x>255] = 255
        x[x<0] = 0
        x = np.array(x, img_target.dtype)
        img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer] = x

    return img_target

print("Done")
import matplotlib.pyplot as plt
try:
    import piq
except:
    os.system("pip3 install piq")
        
from piq import psnr,ssim,multi_scale_ssim
from typing import Union, Tuple
import time
inpaint_iters=1800
inpainted_images_gan = None
image_batch = None
inpainted_images= None
masks = None
ssim_index = torch.zeros(len(masked_loader),12)
psnr_index = torch.zeros(len(masked_loader),12)

for i,data in enumerate(masked_loader,0):
    
    b_size = data[0].size(0)
    image_batch_cpu = unnormalise(data[0]) # unnormalise cpu tensor
    image_batch = data[0].to(device)
    masks_cpu = data[1].unsqueeze(1)
    masks = data[1].to(device).unsqueeze(1)
    masks_inv = torch.ones_like(masks) - masks
    masks_inv_cpu = torch.ones_like(masks_cpu) - masks_cpu
    
    weighted_masks = (torch.nn.functional.conv2d(masks_inv,special_conv,padding=WINDOW_SIZE//2)*masks)/(WINDOW_SIZE*WINDOW_SIZE)

    z_closest = torch.randn(b_size,noise_dim,1,1,device=device,requires_grad=True)
    z_optimizer = torch.optim.Adam([z_closest])
    
    prior_criterior = nn.BCEWithLogitsLoss()
    p_losses=[]
    c_losses=[]
    start_time = time.time()
    for j in range(inpaint_iters):
        z_optimizer.zero_grad()
        fake_images = Gen_eval(z_closest)
        d_output = Dis_eval(fake_images).view(-1)

        prior_loss= 64*64*3*prior_loss_parameter*prior_criterior(d_output,torch.ones(b_size,device=device))

        context_loss = torch.norm(weighted_masks*(fake_images - image_batch),p=1)
        if (j % 50):
            p_losses.append(prior_loss.detach().cpu())
            c_losses.append(context_loss.detach().cpu())
            
        loss = context_loss + prior_loss
        loss.backward()
        z_optimizer.step()
    inpainted_images_gan_cpu = unnormalise(Gen_eval(z_closest.detach()).cpu())
    # returns a torch tensor of images
    mid_time = time.time()
    inpainted_images_cpu = poissonblending_batch(image_batch_cpu,inpainted_images_gan_cpu,masks_inv_cpu)
    inpainted_images = inpainted_images_cpu.to(device)
    ssim_index[i,:] = ssim(inpainted_images,unnormalise(image_batch),data_range=1.,size_average=False).cpu()
    psnr_index[i,:] = psnr(inpainted_images_cpu,image_batch_cpu,data_range=1.,reduction='none')
    end_time = time.time()
    for img,gen_img,gen_poisson,mask in zip(image_batch_cpu,inpainted_images_gan_cpu,inpainted_images_cpu,masks_cpu):
        og = transforms.ToPILImage(mode="RGB")(img)
        inp = transforms.ToPILImage(mode="RGB")(img*mask + (1-mask)*gen_img)
        inpp = transforms.ToPILImage(mode="RGB")(gen_poisson)
        ogm = transforms.ToPILImage(mode="RGB")(img*mask)
        f = plt.figure()
        f.add_subplot(1,4, 1).set_title('Original')
        plt.imshow(og)
        plt.axis('off')
        f.add_subplot(1,4, 2).set_title('Masked')
        plt.imshow(ogm)
        plt.axis('off')
        f.add_subplot(1,4, 3).set_title('Overlay')
        plt.imshow(inp)
        plt.axis('off')
        f.add_subplot(1,4, 4).set_title('Inpainted')
        plt.imshow(inpp)
        plt.axis('off')
        plt.show()
    
plt.plot(c_losses)
plt.plot(p_losses)
plt.legend(["Context loss","Prior Loss"])
plt.show()
plt.plot(p_losses)
plt.legend(["Prior Loss"])
plt.show()

print(ssim_index.size())
psnr_sum = torch.mean(psnr_index,dim=0)
for i in range(4):
    print(torch.mean(psnr_sum[i::4]))
print(ssim_sum)
print(ssim_index)
import matplotlib.pyplot as plt
try:
    import piq
except:
    os.system("pip3 install piq")
        
from piq import psnr,ssim,multi_scale_ssim
from typing import Union, Tuple
import time
inpaint_iters=1800
inpainted_images_gan = None
image_batch = None
inpainted_images= None
masks = None
ssim_index = torch.zeros(len(masked_loader),12)
psnr_index = torch.zeros(len(masked_loader),12)

for i,data in enumerate(masked_loader,0):
    
    b_size = data[0].size(0)
    image_batch_cpu = unnormalise(data[0]) # unnormalise cpu tensor
    image_batch = data[0].to(device)
    masks_cpu = data[1].unsqueeze(1)
    masks = data[1].to(device).unsqueeze(1)
    masks_inv = torch.ones_like(masks) - masks
    masks_inv_cpu = torch.ones_like(masks_cpu) - masks_cpu
    
    weighted_masks = (torch.nn.functional.conv2d(masks_inv,special_conv,padding=WINDOW_SIZE//2)*masks)/(WINDOW_SIZE*WINDOW_SIZE)

    z_closest = torch.randn(b_size,noise_dim,1,1,device=device,requires_grad=True)
    z_optimizer = torch.optim.Adam([z_closest])
    
    prior_criterior = nn.BCEWithLogitsLoss()
    p_losses=[]
    c_losses=[]
    start_time = time.time()
    for j in range(inpaint_iters):
        z_optimizer.zero_grad()
        fake_images = Gen_eval(z_closest)
        d_output = Dis_eval(fake_images).view(-1)

        prior_loss= 64*64*3*prior_loss_parameter*prior_criterior(d_output,torch.ones(b_size,device=device))

        context_loss = torch.norm(weighted_masks*(fake_images - image_batch),p=1)
        if (j % 50):
            p_losses.append(prior_loss.detach().cpu())
            c_losses.append(context_loss.detach().cpu())
            
        loss = context_loss + prior_loss
        loss.backward()
        z_optimizer.step()
    inpainted_images_gan_cpu = unnormalise(Gen_eval(z_closest.detach()).cpu())
    # returns a torch tensor of images
    mid_time = time.time()
    inpainted_images_cpu = poissonblending_batch(image_batch_cpu,inpainted_images_gan_cpu,masks_inv_cpu)
    inpainted_images = inpainted_images_cpu.to(device)
    ssim_index[i,:] = ssim(inpainted_images,unnormalise(image_batch),data_range=1.,size_average=False).cpu()
    psnr_index[i,:] = psnr(inpainted_images_cpu,image_batch_cpu,data_range=1.,reduction='none')
    end_time = time.time()
    for img,gen_img,gen_poisson,mask in zip(image_batch_cpu,inpainted_images_gan_cpu,inpainted_images_cpu,masks_cpu):
        og = transforms.ToPILImage(mode="RGB")(img)
        inp = transforms.ToPILImage(mode="RGB")(img*mask + (1-mask)*gen_img)
        inpp = transforms.ToPILImage(mode="RGB")(gen_poisson)
        ogm = transforms.ToPILImage(mode="RGB")(img*mask)
        f = plt.figure()
        f.add_subplot(1,4, 1).set_title('Original')
        plt.imshow(og)
        plt.axis('off')
        f.add_subplot(1,4, 2).set_title('Masked')
        plt.imshow(ogm)
        plt.axis('off')
        f.add_subplot(1,4, 3).set_title('Overlay')
        plt.imshow(inp)
        plt.axis('off')
        f.add_subplot(1,4, 4).set_title('Inpainted')
        plt.imshow(inpp)
        plt.axis('off')
        plt.show()
    break
        
    
    