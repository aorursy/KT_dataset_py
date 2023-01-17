# This Python 3 environment comes with many helpful analytics libraries installedimport numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import glob

import PIL



import numpy as np

import torch

import torchvision

from torch import nn

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
!ls -l ../input/testfaces/faces/faces/
device = torch.device("cuda:0")

device
class FaceDataset(Dataset):

    def __init__(self, path, transform=None):

        self.path = path

        self.transform = transform #or transforms.ToTensor()

        self.imgs = glob.glob(os.path.join(path, "*.jpg"))

        

    def __len__(self):

        return len(self.imgs)

    

    def __getitem__(self, index):

        return self.transform(PIL.Image.open(self.imgs[index]))
def gaussian_noise(x, std=0.5, weight=0.2):

    noise = torch.zeros_like(x)

    noise.data.normal_(0, std=std)

    return x+weight*noise
transform = transforms.Compose([

    transforms.Resize(128),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),

    transforms.ToTensor()

])



dataset_A = FaceDataset("../input/testfaces/faces/faces/cage", transform=transform)

dataloader_A = torch.utils.data.DataLoader(dataset_A, batch_size=32, shuffle=True)



dataset_B = FaceDataset("../input/testfaces/faces/faces/trump", transform=transform)

dataloader_B = torch.utils.data.DataLoader(dataset_B, batch_size=32, shuffle=True)
class Print(nn.Module):

    def forward(self, input):

        print(input.shape)

        return input

    

class Flatten(nn.Module):

    def forward(self, input):

        output = input.view(input.size(0), -1)

        return output



class Reshape(nn.Module):

    def __init__(self, shape=(256, 5, 5)):

        super().__init__()

        self.shape = shape

        

    def forward(self, input):

        output = input.view(-1, *self.shape)  # channel * 4 * 4

        return output

    

class AutoEncoder(nn.Module):

    def build_encoder(self, out_features):

        return nn.Sequential(

            nn.Conv2d(3, 512, 3, stride=3, padding=1),

            nn.LeakyReLU(0.1, True),

            nn.MaxPool2d(2, stride=2),

            

            nn.Conv2d(512, 256, 3, stride=2, padding=1), 

            nn.LeakyReLU(0.1,True),

            nn.MaxPool2d(2, stride=1),

            

            nn.Conv2d(256, 128, 3, stride=2, padding=1), 

            nn.LeakyReLU(0.1,True),

            #nn.MaxPool2d(2, stride=1),

            #Print(),

            Flatten(),

            nn.Tanh(),

            

            #Print(),

            

            nn.Linear(3200, out_features),

            nn.Sigmoid()

        )

        

    def build_decoder(self, in_features):

        return nn.Sequential(

            nn.Linear(in_features, 3200),

            nn.Tanh(),

            

            Reshape(shape=(128, 5, 5)),

            #Print(),

            

            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=3, padding=0), #16x16 

            nn.LeakyReLU(0.1, True),

            

            nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=1), #32x32 

            nn.LeakyReLU(0.1, True),

            

            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1), #64x64 

            nn.LeakyReLU(0.1, True),

            

            #Print(),

            nn.ConvTranspose2d(512, 3, kernel_size=4, stride=2, padding=1),

            nn.Sigmoid(),

            #Print()

        )

    

    def __init__(self):

        super().__init__()

        self.encoder = self.build_encoder(2048)

        

        features = 1024

        

        self.mean = nn.Linear(2048, features)

        self.std = nn.Linear(2048, features)

        

        self.decoder_A = self.build_decoder(features)

        self.decoder_B = self.build_decoder(features)

    

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5*logvar)

        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    

    def forward(self, x, select='A'):

        #print("Input shape:", x.shape)

        

        x = self.encoder(x)

        #print("Encoded shape:", x.shape)



        mu, logvar = self.mean(x), self.std(x)

        z = self.reparameterize(mu, logvar)

        y = None

        

        if select == 'A':

            y = self.decoder_A(z)

        else:

            y = self.decoder_B(z)

        #print("Decoded shape: ", y.shape)

        

        return y, mu, logvar
class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        

        self.discriminator = nn.Sequential(

            nn.Conv2d(3, 256, kernel_size=3, stride=2),

            nn.ReLU(True),

            

            nn.Conv2d(256, 128, kernel_size=3, stride=2),

            nn.ReLU(True),

            

            nn.Conv2d(128, 64, 3, stride=2),

            nn.ReLU(True),

            

            nn.Conv2d(64, 32, 3, stride=2),

            nn.ReLU(True),

            

            Flatten(),

            #Print(),

            

            nn.Linear(1568, 512),

            nn.ReLU(True),

            

            nn.Linear(512, 128),

            nn.LeakyReLU(0.1, True),

            

            nn.Linear(128, 1),

            nn.Sigmoid(),

        )

        

    def forward(self, x):

        return self.discriminator(x)
def plot_batch(batch, size=(20, 10), adjust=False, nrow=8, savefig=None):

    if adjust:

        batch = batch.clamp(0, 1)

        batch = batch.view(batch.size(0), 3, 128, 128)

    

    image = torchvision.utils.make_grid(batch, nrow=nrow).cpu().detach()

    batch = np.transpose(np.uint8(image.numpy()*255), (1, 2, 0))

    plt.figure(figsize=size)

    plt.imshow(batch)

    plt.show()

    if savefig is not None:

        torchvision.utils.save_image(image, savefig)

    plt.pause(0.001)
batch = next(iter(dataloader_A))

plot_batch(batch)
model = AutoEncoder().to(device)

D_A = Discriminator().to(device)

D_B = Discriminator().to(device)
model.load_state_dict(torch.load("../input/faceswap-trump-in-a-cage/autoencoder.pt"))

D_A.load_state_dict(torch.load("../input/faceswap-trump-in-a-cage/discriminator_A.pt"))

D_B.load_state_dict(torch.load("../input/faceswap-trump-in-a-cage/discriminator_B.pt"))
class KLDLoss(nn.Module):

    def forward(self, mu, logvar):

        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
import torch.nn.functional as F



reconstruction_loss = F.binary_cross_entropy

kld_loss = KLDLoss()

criterion = lambda y, x, mu, logvar: reconstruction_loss(y, x, reduction='sum') + kld_loss(mu, logvar)



optimizer_A = torch.optim.Adam(

    [{'params': model.encoder.parameters()}, {'params': model.std.parameters()}, {'params': model.mean.parameters()}, {'params': model.decoder_A.parameters()}], 

    lr=5e-5,betas=(0.5, 0.999)

)

optimizer_B = torch.optim.Adam(

    [{'params': model.encoder.parameters()}, {'params': model.std.parameters()}, {'params': model.mean.parameters()}, {'params': model.decoder_B.parameters()}], 

    lr=5e-5, betas=(0.5, 0.999)

)



criterion_D = nn.BCELoss() #nn.CrossEntropyLoss()

optimizer_D_A = torch.optim.Adam(D_A.discriminator.parameters(), lr=5e-5)

optimizer_D_B = torch.optim.Adam(D_B.discriminator.parameters(), lr=5e-5)
batch = next(iter(dataloader_A))

y = model(batch.to(device))

plot_batch(batch)

plot_batch(y[0], adjust=True)
def train_discriminator(D, criterion, optimizer, real, fake):

    optimizer.zero_grad()

    

    with torch.set_grad_enabled(True):

        pred_real = D(real)

        pred_fake = D(fake)

        

        loss_real = criterion(pred_real, torch.ones(real.size(0), 1).to(device))

        loss_fake = criterion(pred_fake, torch.zeros(fake.size(0), 1).to(device))

        

        loss = loss_real + loss_fake

        loss.backward(retain_graph=True)



        optimizer.step()

    

    return loss.item()



def train_generator(G, D, criterion_G, criterion_D, optimizer, x, fake, mu, logvar):

    optimizer.zero_grad()

    

    with torch.set_grad_enabled(True):

        prediction = D(fake)

        

        target = torch.ones(x.size(0), 1).to(device)

        d_loss = criterion_D(prediction, target)

        d_loss.backward(retain_graph=True)

        d_loss = d_loss.item()

        

        if criterion_G is not None:

            g_loss = criterion_G(fake, x, mu, logvar)

            grads = torch.ones_like(g_loss)

            g_loss.backward(grads, retain_graph=True)

            d_loss += g_loss.mean().item()

            

        optimizer.step()

    

    return d_loss
from IPython.display import clear_output



num_epochs = 500

model.train()

D_A.train()

D_B.train()



loss_hist = {'D': list(), 'G': list()}



for epoch in range(num_epochs): 

    num_batch = 0

    

    g_loss = 0

    d_loss = 0



    for x_A, x_B in zip(iter(dataloader_A), iter(dataloader_B)):

        x_A = x_A.to(device)

        x_B = x_B.to(device)



        #Normal reconstruction

        fake_A, mu_A, logvar_A = model(x_A)

        fake_B, mu_B, logvar_B = model(x_B, select='B')

        

        #Train discriminator on normal faces

        d_loss += train_discriminator(D_A, criterion_D, optimizer_D_A, x_A, fake_A)

        d_loss += train_discriminator(D_B, criterion_D, optimizer_D_B, x_B, fake_B)

        

        #Train Generator on normal faces

        g_loss += train_generator(model, D_A, criterion, criterion_D, optimizer_A, x_A, fake_A, mu_A, logvar_A)

        g_loss += train_generator(model, D_B, criterion, criterion_D, optimizer_B, x_B, fake_B, mu_B, logvar_B)

        

        #Reconstruction with swapped faces

        fake_A2, mu_A2, logvar_A2 = model(x_A, select='B')

        fake_B2, mu_B2, logvar_B2 = model(x_B, select='A')

        

        #Adversarial only Training for faceswapping

        g_loss += train_generator(model, D_B, None, criterion_D, optimizer_B, x_A, fake_A2, mu_A2, logvar_A2)

        g_loss += train_generator(model, D_A, None, criterion_D, optimizer_A, x_B, fake_B2, mu_B2, logvar_B2)



        num_batch += 1

    

    d_loss /= num_batch*2

    g_loss /= num_batch*4

    

    loss_hist['D'].append(d_loss)

    loss_hist['G'].append(g_loss)

    

    print("\rEpoch {}/{}, Avg. D-loss: {:.4f}, Avg. G-Loss: {:.4f}".format(epoch+1, num_epochs, d_loss, g_loss), end='', flush=True)

    

    if epoch % 100 == 0:

        clear_output()

        

        x_A = next(iter(dataloader_A)).to(device)

        x_B = next(iter(dataloader_B)).to(device)

        plot_batch(torch.cat((x_A, x_B, model(x_A)[0], model(x_B, select='B')[0])), size=(30, 15), adjust=True, nrow=16)



torch.save(model.state_dict(), "autoencoder.pt")

torch.save(D_A.state_dict(), "discriminator_A.pt")

torch.save(D_B.state_dict(), "discriminator_B.pt")
plt.figure(figsize=(25, 5))

plt.plot(loss_hist['D'], label='D-Loss')

plt.plot(loss_hist['G'], '--', label='G-Loss')

plt.legend()

plt.savefig('VAEGAN-Loss.png')

plt.show()
x_A = next(iter(dataloader_A)).to(device)

x_B = next(iter(dataloader_B)).to(device)



plot_batch(torch.cat((x_A, x_B, model(x_A)[0], model(x_B, select='B')[0])), size=(30, 15), adjust=True, nrow=16, savefig='reconstruction.png')
batch = next(iter(dataloader_A)).to(device)

y_B = model(batch, select='B')



plot_batch(torch.cat((batch, y_B[0])), adjust=True, nrow=16, savefig='cagetrump.png')
batch = next(iter(dataloader_B)).to(device)

y = model(batch)

plot_batch(torch.cat((batch, y[0])), adjust=True, nrow=16, savefig='trumpcage.png')