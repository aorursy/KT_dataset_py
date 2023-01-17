import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# set random seed
manual_seed = 999
random.seed(manual_seed)
torch.manual_seed(manual_seed)

batch_size = 32
image_size = 64
channels = 3
z_dim = 100
ngpu = 1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
trans = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_set = dset.ImageFolder(root='../input/jovian-pytorch-z2g/Human protein atlas/', transform=trans)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

print(len(train_set))

batch = next(iter(train_loader))[0]

nrows=4
ncols=4
fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=True)
for i in range(nrows):
    for j in range(ncols):
        img = batch[nrows * i + j].numpy()
        img = img.transpose((1, 2, 0))
        axes[i, j].imshow(img)

plt.show()
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.network = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # state size = 512*4*4
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # state size = 256*8*8
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # state size = 128*16*16
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # state size = 64*32*32
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # state size = 3*64*64
        )
    
    def forward(self, x):
        return self.network(x)

# create the generator
netG = Generator().to(device)

if device.type == 'cuda':
    netG.to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda' and ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# apply the weight_init function to randomly initialize all the weights
netG.apply(weight_init)

# print the model
print(netG)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size = 128*32*32
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size = 128*16*16
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size = 256*8*8
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size = 512*4*4
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # state size 1*1*1
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

netD = Discriminator().to(device)

if device.type == 'cuda' and ngpu > 1:
    netD = nn.DataParallel(netD, list(range(ngpu)))

# initializing the weights
netD.apply(weight_init)

print(netD)
# inititalize the BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
# the progression of the generator
fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_labels = 1
fake_labels = 0

# Setup Adam optimizers for both G and D
lr = 0.0002

g_optim = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
d_optim = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

# lists to keep track of progress

img_list = []
g_losses = []
d_losses = []
iters = 2
epochs = 3
print_every = 200
save_img_every = 10

# training loop
print('starting training...')

for epoch in range(epochs):
    for i, data in enumerate(train_loader):
        
        ################################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))  #
        ################################################################
        
        # train with all-real batch
        netD.zero_grad()
        # Format batch
        real = data[0].to(device)
        b_size = real.size(0)
        label = torch.full((b_size,), real_labels, device=device)
        
        # forward pass real batch through D
        output = netD(real).view(-1)
        
        # calculate loss on all-real batch
        d_loss_real = criterion(output, label)
        
        # calculate gradients for D in backward pass
        d_loss_real.backward()
        d_x = output.mean().item()
        
        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, z_dim, 1, 1, device=device)
        # generate fake images with G
        fake = netG(noise)
        label.fill_(fake_labels)
        # classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # calculate D's loss on the all-fake batch
        d_loss_fake = criterion(output, label)
        # Calculate the gradients for this batch
        d_loss_fake.backward()
        d_g_z1 = output.mean().item()
        # add the gradients from the all-real and all-fake batches
        d_loss = d_loss_fake + d_loss_real
        # update D
        d_optim.step()
        
        ################################################
        # (2) Update G network: maximize log(D(G(z)))  #
        ################################################
        
        netG.zero_grad()
        label.fill_(real_labels)  # fake labels are real for generator cost
        
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        
        # Calculate G's loss based on this output
        g_loss = criterion(output, label)
        # Calculate gradients for G
        g_loss.backward()
        d_g_z2 = g_loss.mean().item()
        # Update G
        g_optim.step()
        
        if i % print_every == 0:
            print('[{}/{}][{}/{}]\tLoss_D: {:.4f}\tLoss_G: {:.4f}\tD(x): {:.4f}\tD(G(z)): {:.4f}'.format(
                epoch, epochs, i, len(train_loader), d_loss.item(), g_loss.item(), d_x, d_g_z1, d_g_z2))
        
        # save losses for plotting
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        
        # Output training stats
        if i % save_img_every == 0:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        iters += 1

print('end of training...')

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_losses,label="G")
plt.plot(d_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())
# Grab a batch of real images from the dataloader
real_batch = next(iter(train_loader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
