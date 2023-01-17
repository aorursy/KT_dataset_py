!ls /kaggle/input/gan-getting-started

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
import torch.utils.model_zoo as model_zoo
from torch import nn
import wandb

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch.optim as optim
import torch
from torch import nn
from torch.nn import functional as F
import pdb
import glob
import tqdm
import wandb
! pip install wandb --upgrade
!wandb login
run = wandb.init(project="painter", config=dict(
  IMAGE_HEIGHT = 256,
  IMAGE_WIDTH  = 256,
  batch_size = 4,          # input batch size for training (default: 64)
  train_batch_size = 4,
  test_batch_size = 10,    # input batch size for testing (default: 1000)
  epochs = 25,            # number of epochs to train (default: 10)
  lr=2e-4,
  betas=(0.5, 0.999),
  momentum = 0.1,      # SGD momentum (default: 0.5) 
  no_cuda = False,         # disables CUDA training
  seed = 42,        # random seed (default: 42)
  log_interval = 50     # how many batches to wait before logging training status
))
run.config.IMAGE_HEIGHT
class ITIDataset(Dataset):
    """Image to Image dataset."""

    def __init__(self, transform=None):
      self.monet_files = glob.glob("/kaggle/input/gan-getting-started/monet_jpg/*.jpg")
      self.photo_files = glob.glob("/kaggle/input/gan-getting-started/photo_jpg/*.jpg")
      self.transform = transform
      
    def __len__(self):
        return max(len(self.monet_files), len(self.photo_files))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
                
        photo_img = self.photo_files[idx]
        monet_img = self.monet_files[int(np.random.uniform(0, len(self.monet_files)))]

        photo_img = Image.open(photo_img)
        monet_img = Image.open(monet_img)
        
        if self.transform:
          photo_img = self.transform(photo_img) 
          monet_img = self.transform(monet_img)
        
        return photo_img, monet_img
data_transform = transforms.Compose([
        transforms.Resize((run.config.IMAGE_HEIGHT, run.config.IMAGE_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),        
    ])
iti_train_dataset    = ITIDataset(transform=data_transform)
train_dataset_loader = DataLoader(iti_train_dataset, batch_size=run.config.train_batch_size, shuffle=True, num_workers=4)
iti_test_dataset      = [iti_train_dataset[int(np.random.uniform(0, len(iti_train_dataset)))] for _ in range(len(iti_train_dataset)//run.config.test_batch_size)]
test_dataset_loader = DataLoader(iti_test_dataset, batch_size=run.config.test_batch_size, shuffle=True, num_workers=4)
for data in train_dataset_loader:
  break
data[0][1].permute
plt.imshow(data[1][3].permute(1,2,0))
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
class Downsample(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding, apply_normalization=True):
    super().__init__()
    self.apply_normalization = apply_normalization
    self.conv_1      = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding=padding,
                                 kernel_size=kernel_size, stride=2, bias=False)
    
    nn.init.kaiming_normal_(self.conv_1.weight, mode="fan_in")
    if self.apply_normalization:
      self.normalizer  = nn.InstanceNorm2d(out_channels, affine=True)
    self.attn        = nn.ReLU()
    
  def forward(self, x):
    x = self.conv_1(x)
    if self.apply_normalization:
      x = self.normalizer(x)
    x = self.attn(x)
    return x
class Upsample(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding, apply_dropout=False):
    super().__init__()
    self.apply_dropout = apply_dropout
    self.conv_1      = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, padding=padding,
                                 kernel_size=kernel_size, stride=2, bias=False)
    
    nn.init.kaiming_normal_(self.conv_1.weight, mode="fan_in")
    self.normalizer  = nn.InstanceNorm2d(out_channels, affine=True)
    if self.apply_dropout:
      self.dropout = nn.Dropout2d(0.2)
    self.attn        = nn.ReLU()
    
  def forward(self, x):
    x = self.conv_1(x)
    x = self.normalizer(x)
    if self.apply_dropout:
      x = self.dropout(x)
    x = self.attn(x)
    return x
class Generator(nn.Module):  
  def __init__(self, device):
    super().__init__()

    self.down_stack = nn.Sequential(
        Downsample(3, 64, 4, padding=(1,1), apply_normalization=False), # (bs, 64, 128, 128)
        Downsample(64, 128, 4, padding=(1,1)), # (bs, 128, 64, 64)
        Downsample(128, 256, 4, padding=(1,1)), # (bs, 256, 32, 32)
        Downsample(256, 512, 4, padding=(1,1)), # (bs, 512, 16, 16)
        Downsample(512, 512, 4, padding=(1,1)), # (bs, 512, 8, 8)
        Downsample(512, 512, 4, padding=(1,1)), # (bs, 512, 4, 4)
        Downsample(512, 512, 4, padding=(1,1)), # (bs, 512, 2, 2)
    )

    self.up_stack = nn.Sequential(
        Upsample(512, 512, 4, apply_dropout=True, padding=(1,1)), # (bs, 2, 2, 1024)
        Upsample(1024, 512, 4, apply_dropout=True, padding=(1,1)), # (bs, 4, 4, 1024)
        Upsample(1024, 512, 4, apply_dropout=True, padding=(1,1)), # (bs, 8, 8, 1024)
        Upsample(1024, 256, 4, padding=(1,1)), # (bs, 16, 16, 1024)
        Upsample(512, 128, 4, padding=(1,1)), # (bs, 32, 32, 512)
        Upsample(256, 64, 4, padding=(1,1)), # (bs, 64, 64, 256)
        # Upsample(128, 64, 4, padding=(1,1)), # (bs, 128, 128, 128)
    )
    self.conv_out    = nn.ConvTranspose2d(128, 3, padding=(1,1),
                                 kernel_size=4, stride=2, bias=False)


  def forward(self, x):
    skips = []
    # pdb.set_trace()
    for down in self.down_stack:
      x = down(x)
      skips.append(x)
    # for skip in skips:
    #   print(skip.shape)
    # print(len(skips))
    skips = reversed(skips[:-1])
    # pdb.set_trace()
    # print(len([skips]))
    for up, skip in zip(self.up_stack, skips):
      x = up(x)
      x = torch.cat((x, skip), 1)
      # print(x.shape)

    x = self.conv_out(x)

    return x

gen = Generator(device).to(device)
gen(torch.rand(2,3,256,256).to(device)).shape
print(len([1,2,3,4,5,6,7]))
class Discriminator(nn.Module):
    def __init__(self,device):
      super().__init__()

      self.down = nn.Sequential(
                   Downsample(3, 64, 4, (1, 1), apply_normalization=False),
                   Downsample(64, 128,  4, (1, 1)),
                   Downsample(128, 256,  4, (1, 1)),
      )
      self.zero_pad = nn.ZeroPad2d(2)

      self.conv1    = nn.Conv2d(256, 512, kernel_size=4, stride=1, bias=False)
      self.normalizer  = nn.InstanceNorm2d(512, affine=True)
      self.conv2     = nn.Conv2d(512, 1, 4, stride=1)
      self.relu      = nn.LeakyReLU()
    def forward(self, x):
      # pdb.set_trace()

      for down in self.down:
        x = down(x)
      x = self.zero_pad(x)
      x = self.conv1(x)
      x = self.normalizer(x)
      x = self.relu(x)
      x = self.conv2(x)
      return x
class DiscriminatorLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.real_loss = nn.BCEWithLogitsLoss(reduction="mean")
    self.generated_loss = nn.BCEWithLogitsLoss(reduction="mean")

  def forward(self, real, generated):
    return 0.5*(self.real_loss(real, torch.ones_like(real)) \
                + self.generated_loss(generated, torch.zeros_like(generated)))
  # total_disc_loss = real_loss + generated_loss

  # return total_disc_loss * 0.5
class GeneratedLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.generated_loss = nn.BCEWithLogitsLoss(reduction="mean")
  def forward(self, generated):
    return self.generated_loss(generated, torch.ones_like(generated))
class CycleLoss(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, real_image, cycled_image, lam):
    loss1 = torch.mean(torch.abs_(real_image - cycled_image))
    return lam * loss1
class IdentityLoss(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, real_image, same_image, lam):
    loss1 = torch.mean(torch.abs_(real_image - same_image))
    return lam * 0.5 * loss1
class GAN(nn.Module):
  def __init__(self):
    super().__init__()

    self.monet_generator = Generator(device).to(device)
    self.monet_discriminator = Discriminator(device).to(device)
    self.photo_generator = Generator(device).to(device)
    self.photo_discriminator = Discriminator(device).to(device)
  def forward(self,real_monet, real_photo):
    # photo -> monet -> photo
    fake_monet   = self.monet_generator(real_photo)
    cycled_photo = self.photo_generator(fake_monet)

    #monet -> photo -> monet
    fake_photo   = self.photo_generator(real_monet)
    cycled_monet = self.monet_generator(fake_photo)

    #generate itself
    same_monet = self.monet_generator(real_monet)
    same_photo = self.photo_generator(real_photo)

    #discriminator in real images
    disc_real_monet = self.monet_discriminator(real_monet)
    disc_real_photo = self.photo_discriminator(real_photo)  

    #discriminator for fake output
    disc_fake_monet = self.monet_discriminator(fake_monet)
    disc_fake_photo = self.photo_discriminator(fake_photo)

    return cycled_photo, cycled_monet, same_monet, same_photo, disc_real_monet, disc_real_photo, disc_fake_monet, disc_fake_photo, fake_monet, fake_photo
def train_step(model, 
              monet_generator_optimizer,
              photo_generator_optimizer,
              monet_discriminator_optimizer,
              photo_discriminator_optimizer,
              discriminator_loss,
              generator_loss,
              cycle_loss,
              identity_loss,
              train_dataloader, 
              device,
              epoch,
              lambda_cycle=10):
  
  model.train()

  for idx, (real_photo, real_monet) in enumerate(tqdm.notebook.tqdm_notebook(train_dataloader, desc='Training epoch ' + str(epoch + 1) + '')):
    real_monet, real_photo = real_monet.to(device), real_photo.to(device)
    
    cycled_photo, cycled_monet, same_monet, same_photo, disc_real_monet, disc_real_photo, disc_fake_monet, disc_fake_photo, _, _ = model(real_monet, real_photo)
    
    #evaluate generateor loss
    monet_gen_loss = generator_loss(disc_fake_monet)
    photo_gen_loss = generator_loss(disc_fake_photo)

    # evaluates total cycle consistency loss
    total_cycle_loss = cycle_loss(real_monet, cycled_monet, lambda_cycle) + cycle_loss(real_photo, cycled_photo, lambda_cycle)

    # evaluates total generator loss
    total_monet_gen_loss = monet_gen_loss + total_cycle_loss + identity_loss(real_monet, same_monet, lambda_cycle)
    total_photo_gen_loss = photo_gen_loss + total_cycle_loss + identity_loss(real_photo, same_photo, lambda_cycle)

    # evaluates discriminator loss
    monet_disc_loss = discriminator_loss(disc_real_monet, disc_fake_monet)
    photo_disc_loss = discriminator_loss(disc_real_photo, disc_fake_photo)

    monet_generator_optimizer.zero_grad()
    photo_generator_optimizer.zero_grad()
    monet_discriminator_optimizer.zero_grad()
    photo_discriminator_optimizer.zero_grad()

    total_monet_gen_loss.backward(retain_graph=True)
    total_photo_gen_loss.backward(retain_graph=True)
    monet_disc_loss.backward(retain_graph=True)
    photo_disc_loss.backward(retain_graph=True)

    monet_generator_optimizer.step()
    photo_generator_optimizer.step()
    monet_discriminator_optimizer.step()
    photo_discriminator_optimizer.step()

    if idx%run.config.log_interval == 0:
      print({
          "monet_gen_loss": total_monet_gen_loss,
          "photo_gen_loss": total_photo_gen_loss,
          "monet_disc_loss": monet_disc_loss,
          "photo_disc_loss": photo_disc_loss
      })
def test(model,
        discriminator_loss,
        generator_loss,
        cycle_loss,
        identity_loss,
        test_dataloader, 
        device,
        epoch,
        lambda_cycle=10):
    # Switch model to evaluation mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
    model.eval()

    monet_gen_loss = 0
    photo_gen_loss = 0
    total_cycle_loss = 0
    total_monet_gen_loss = 0
    total_photo_gen_loss = 0
    monet_disc_loss = 0
    photo_disc_loss = 0
    
    real_photo_log = []
    converted_photo_log = []
    real_monet_log = []
    converted_monet_log = []
    with torch.no_grad():
        # for real_monet, real_photo in test_dataloader:
        for idx, (real_photo, real_monet) in enumerate(tqdm.notebook.tqdm_notebook(test_dataloader, desc='Test epoch ' + str(epoch + 1) + '')):

            real_monet, real_photo = real_monet.to(device), real_photo.to(device)
            
            cycled_photo, cycled_monet, same_monet, same_photo, disc_real_monet, disc_real_photo, disc_fake_monet, disc_fake_photo, fake_monet, fake_photo= model(real_monet, real_photo)
    
            #evaluate generateor loss
            monet_gen_loss += generator_loss(disc_fake_monet)
            photo_gen_loss += generator_loss(disc_fake_photo)

            # evaluates total cycle consistency loss
            total_cycle_loss += cycle_loss(real_monet, cycled_monet, lambda_cycle) + cycle_loss(real_photo, cycled_photo, lambda_cycle)

            # evaluates total generator loss
            total_monet_gen_loss += monet_gen_loss + total_cycle_loss + identity_loss(real_monet, same_monet, lambda_cycle)
            total_photo_gen_loss += photo_gen_loss + total_cycle_loss + identity_loss(real_photo, same_photo, lambda_cycle)

            # evaluates discriminator loss
            monet_disc_loss += discriminator_loss(disc_real_monet, disc_fake_monet)
            photo_disc_loss += discriminator_loss(disc_real_photo, disc_fake_photo)
            
            # WandB – Log images in your test dataset automatically, along with predicted and true labels by passing pytorch tensors with image data into wandb.Image
            # real_photo_log.append(wandb.Image(
            #     real_monet[0], caption="Real Photo"))
            
            # converted_photo.append(wandb.Image(
            #     cycled_photo[0], caption="Converted Photo"))

            # real_monet.append(wandb.Image(
            #     real_monet[0], caption="Real Monet"))
            
            # converted_monet.append(wandb.Image(
            #     cycled_monet[0], caption="Converted Photo"))
            
    # WandB – wandb.log(a_dict) logs the keys and values of the dictionary passed in and associates the values with a step.
    # You can log anything by passing it to wandb.log, including histograms, custom matplotorch.rand(1,torch.rand(1,torch.rand(1,torch.rand(1,3,128,128)3,128,128)3,128,128)3,128,128)tlib objects, images, video, text, tables, html, pointclouds and other 3D objects.
    # Here we use it to log test accuracy, loss and some test images (along with their true and predicted labels).
        wandb.log({
            "Fake Photo": wandb.Image(
                fake_photo[0], caption="Fake Photo"),
            "Fake Monet": wandb.Image(
                fake_monet[0], caption="Fake Monet"),
            "epoch": epoch + 1,
            "monet_gen_loss": monet_gen_loss / len(test_dataloader.dataset),
            "photo_gen_loss": photo_gen_loss / len(test_dataloader.dataset),
            "total_cycle_loss":total_cycle_loss / len(test_dataloader.dataset),
            "total_monet_gen_loss":total_monet_gen_loss / len(test_dataloader.dataset),
            "total_photo_gen_loss":total_photo_gen_loss / len(test_dataloader.dataset),
            "monet_disc_loss":monet_disc_loss / len(test_dataloader.dataset),
            "photo_disc_loss":photo_disc_loss / len(test_dataloader.dataset),
        })
if __name__ == "__main__":
    model = GAN().to(device)

    monet_generator_optimizer = torch.optim.Adam(model.monet_generator.parameters(), lr=run.config.lr, betas=run.config.betas)
    photo_generator_optimizer = torch.optim.Adam(model.photo_generator.parameters(), lr=run.config.lr, betas=run.config.betas)

    monet_discriminator_optimizer = torch.optim.Adam(model.monet_discriminator.parameters(), lr=run.config.lr, betas=run.config.betas)
    photo_discriminator_optimizer = torch.optim.Adam(model.photo_discriminator.parameters(), lr=run.config.lr, betas=run.config.betas)

    discriminator_loss = DiscriminatorLoss().to(device)
    generator_loss     = GeneratedLoss().to(device)
    cycle_loss         = CycleLoss().to(device)
    identity_loss      = IdentityLoss().to(device)


#     weights_file = wandb.restore('model.h5')
    # use the "name" attribute of the returned object
    # if your framework expects a filename, e.g. as in Keras
    # model.load_state_dict(torch.load(PATH))

#     model.load_state_dict(torch.load(weights_file.name))
    wandb.watch([model], log="all")

    for epoch in range(run.config.epochs):
        train_step(model, 
                  monet_generator_optimizer,
                  photo_generator_optimizer,
                  monet_discriminator_optimizer,
                  photo_discriminator_optimizer,
                  discriminator_loss,
                  generator_loss,
                  cycle_loss,
                  identity_loss,
                  train_dataset_loader, 
                  device,
                  epoch,
                  lambda_cycle=10)
        # if epoch%run.config.log_interval == 0:
        test(model,
          discriminator_loss,
          generator_loss,
          cycle_loss,
          identity_loss,
          test_dataset_loader, 
          device,
          epoch,
          lambda_cycle=10)

        torch.save(model.state_dict(), "model.h5")
        wandb.save('model.h5')
  # train_step()
_, ax = plt.subplots(5, 2, figsize=(10, 10))


for i in range(5):
    data = next(iter(dataset_loader))

    img = (data[0][0]).permute(1,2,0).cpu().detach().numpy()

    prediction = m_gen(data[0].to(device))[0].permute(1,2,0).cpu().detach().numpy()

    # prediction = (prediction).astype(np.uint8)
    # img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

    ax[i, 0].imshow(img)
    ax[i, 1].imshow(prediction)
    ax[i, 0].set_title("Input Photo")
    ax[i, 1].set_title("Monet-esque")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")
plt.show()

data = next(iter(dataset_loader))
data[1].shape


