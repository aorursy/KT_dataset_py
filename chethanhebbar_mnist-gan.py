# importing the necessary libraries

import numpy as np
import os
import torch 
import torchvision
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.datasets import MNIST
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
%matplotlib inline
# defining the image size and the batch size
image_size = 28
batch_size = 69
def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(images.detach()[:nmax], nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break
# creating the training and testing datasets
train_ds = MNIST(root = "data/", train = True, transform = T.ToTensor(), download = True)
test_ds = MNIST(root = "data/", train = False, transform = T.ToTensor())

# looking at some of the images from the dataset
image, label = train_ds[0]
plt.imshow(image[0], cmap='gray')
print('Label:', label)
# printing out the shape of these images and labels
print(train_ds[0][0].shape)
# so its an image of pixel size 28x28
def get_default_device():
    # Pick gpu if available else cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    else:
        return torch.device("cpu")
    

def to_device(data, device):
    # move data to the gpu
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    
    return data.to(device, non_blocking = True)

class DeviceDataLoader():
    # Wraps a data loader to move the data to the device
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        # yield a batch of data after moving it to device
        for b in self.dl:
            yield to_device(b, self.device)
            
    def __len__(self):
        # number of batches
        return len(self.dl)
device = get_default_device()
print(device)
# creating the data loaders
train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 3, pin_memory = True)
# transferring the dataloader to the gpu
train_dl = DeviceDataLoader(train_dl, device)
discriminator = nn.Sequential(
    # input : 1 x 28 x 28
    # instead of max pooling, we are using stride = 2 as it gives better results for gans
    nn.Conv2d(1, 16, kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(16),
    # leaky relu gives alpha x input for input < 0(alpha = 0.2 here)
    nn.LeakyReLU(0.2, inplace = True),
    
    # input : 16 x 14 x 14
    nn.Conv2d(16, 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace = True),
    
    # input : 64 x 7 x 7
    nn.Conv2d(64, 256, kernel_size = 4, stride = 1, padding = 0, bias = False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace = True),
    
    # input : 256 x 4 x 4
    nn.Conv2d(256, 1, kernel_size = 4, stride = 1, padding = 0, bias = False),
    
    # input : 1 x 1 x 1
    nn.Flatten(),
    nn.Sigmoid())
# lets move the discriminator to the gpu
discriminator = to_device(discriminator, device)
# generator performs deconvolution in order to transform a random latent tensor into an image 
# as batch size = 69 we will pass this as the batch size here too
latent_size = 69

generator = nn.Sequential(
    # initially it is a random tensor of 69 x 1 x 1
    nn.ConvTranspose2d(latent_size, 256, kernel_size = 4, stride = 1, padding = 0, bias = False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    
    # input : 256 x 4 x 4
    nn.ConvTranspose2d(256, 64, kernel_size = 4, stride = 1, padding = 0, bias = False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    
    # input : 64 x 7 x 7
    nn.ConvTranspose2d(64, 16, kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.BatchNorm2d(16),
    nn.ReLU(True),
    
    # input : 16 x 14 x 14
    nn.ConvTranspose2d(16, 1, kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.Tanh(),
    # output : 1 x 28 x 28
)
# generating random images
xb = torch.randn(batch_size, latent_size, 1, 1) # random latent tensors
fake_images = generator(xb)
print(fake_images.shape)
# lets move the generator to the gpu
generator = to_device(generator, device)
def train_discriminator(real_images, opt_d):
    
    # clearing any left over gradients in the discriminator
    opt_d.zero_grad()
    
    # pass real images through discriminator
    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device = device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()
    
    # generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device = device)
    fake_images = generator(latent)
    
    # trying to fool the discriminator by passing fake images
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()
    
    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score
def train_generator(opt_g):
    
    # Clear generator gradients
    opt_g.zero_grad()
    
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)
    
    # Try to fool the discriminator
    preds = discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(preds, targets)
    
    # Update generator weights
    loss.backward()
    opt_g.step()
    
    return loss.item()
from torchvision.utils import save_image
sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)
# save samples
def save_samples(index, latent_tensors, show=True):
    
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(fake_images, os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
save_samples(0, fixed_latent)
def fit(epochs, lr, start_idx = 1):
    
    torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for real_images, _ in tqdm(train_dl):
            
            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
            
            # Train generator
            loss_g = train_generator(opt_g)
            
        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
    
        # Save generated images
        save_samples(epoch+start_idx, fixed_latent, show=False)
    
    return losses_g, losses_d, real_scores, fake_scores
lr = 0.0002
epochs = 25
history = fit(epochs, lr)
losses_g, losses_d, real_scores, fake_scores = history
# Save the model checkpoints 
torch.save(generator.state_dict(), 'G.ckpt')
torch.save(discriminator.state_dict(), 'D.ckpt')
from IPython.display import Image
Image('./generated/generated-images-0001.png')
Image('./generated/generated-images-0005.png')
Image('./generated/generated-images-0010.png')
Image('./generated/generated-images-0015.png')
Image('./generated/generated-images-0020.png')
Image('./generated/generated-images-0025.png')
# plotting the losses against epochs
plt.plot(losses_d, '-')
plt.plot(losses_g, '-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Discriminator', 'Generator'])
plt.title('Losses');
# plotting the scores against the epochs
plt.plot(real_scores, '-')
plt.plot(fake_scores, '-')
plt.xlabel('epoch')
plt.ylabel('score')
plt.legend(['Real', 'Fake'])
plt.title('Scores');
