!apt install -y ffmpeg  # for GIF creation

!pip install ffmpeg     # P.S needs internet on
import torch

import torchvision.datasets  as dsets

import torch.utils.data

import torchvision.transforms as transforms

import torch.nn as nn

import torch.nn.functional as F

import torchvision.utils as vutils



import tqdm

import matplotlib.pyplot as plt

import numpy as np



from matplotlib import animation

from matplotlib.animation import FuncAnimation

from IPython.display import HTML, display, FileLink
batch_size = 32

image_size = 64

num_epochs = 2

torch.cuda.set_device("cuda:0")
dset = dsets.ImageFolder(root="/kaggle/input/",

                           transform=transforms.Compose([

                               transforms.Resize(image_size),

                               transforms.CenterCrop(image_size),

                               transforms.ToTensor(),

                           ]))

# cut the size of the dataset

dataset, _ = torch.utils.data.random_split(dset, [len(dset)//2, len(dset)-len(dset)//2])

del _

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)
real_batch = next(iter(dataloader))

plt.figure(figsize=(8, 8))

plt.axis("off")

plt.imshow(np.transpose(vutils.make_grid(real_batch[0], padding=2).cpu(),(1,2,0)))
class Flatten(nn.Module):

    def forward(self, input):

        return input.view(input.size(0), -1)





class UnFlatten(nn.Module):

    def forward(self, input, size=1024):

        return input.view(input.size(0), size, 1, 1)
class VAE(nn.Module):

    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):

        super().__init__()

        

        self.encoder = nn.Sequential(

            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),

            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),

            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2),

            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2),

            nn.ReLU(),

            Flatten()

        )

        

        self.h2mu = nn.Linear(h_dim, z_dim)

        self.h2sigma = nn.Linear(h_dim, z_dim)

        self.z2h = nn.Linear(z_dim, h_dim)

        

        self.decoder = nn.Sequential(

            UnFlatten(),

            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),

            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),

            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),

            nn.ReLU(),

            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),

            nn.Sigmoid(),

        )

        

    def reparameterize(self, mu, logvar):

        std = logvar.mul(0.5).exp_()

        eps = torch.randn(*mu.size())

        z = mu + std * eps

        return z

    

    def bottleneck(self, h):

        mu = self.h2mu(h)

        logvar = self.h2sigma(h)

        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

        

    def encode(self, x):

        return self.bottleneck(self.encoder(x))[0]



    def decode(self, z):

        return self.decoder(self.z2h(z))

    

    def forward(self, x):

        h = self.encoder(x)

        z, mu, logvar = self.bottleneck(h)

        z = self.z2h(z)

        return self.decoder(z), mu, logvar
def vae_loss(recon_x, x, mu, logvar) -> float:

    BCE = F.binary_cross_entropy(recon_x.view(-1, image_size*image_size*3),

                                 x.view(-1, image_size*image_size*3), reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
model = VAE()

try:

    model.load_state_dict(torch.load('vae.pth'))

except:

    print("Weights not found ):")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(num_epochs):

    train_loss = 0

    for data, _ in tqdm.tqdm(dataloader):

        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)

        loss = vae_loss(recon_batch, data, mu, logvar)

        loss.backward()

        train_loss += loss.item()

        optimizer.step()

        

    torch.save(model.state_dict(), "vae.pth")

    print('epoch %d, loss %.4f' % (epoch, train_loss / len(dataset)))

model.eval()

FileLink(r'vae.pth')
def get(x):

    return dataset[x][0].view(1, 3, image_size, image_size)



def imshow(img):

    pic = np.transpose(img.numpy(), (1,2,0))

    plt.axis('off')

    return plt.imshow(pic, animated=True)



def morph(inputs, steps: int, delay: int):

    latent = [model.encode(get(k)).data for k in inputs]

    fig = plt.figure()

    images = []

    for a, b in zip(latent, latent[1:] + [latent[0]]):

        for t in np.linspace(0, 1, steps):

            c = a*(1-t)+b*t

            morphed = model.decode(c).data[0]

            images.append([imshow(morphed)])

    

    ani = animation.ArtistAnimation(fig, images, interval=delay)



    display(HTML(ani.to_html5_video()))
num_images = 30 # amount of images in GIF

num_steps = 20 # smoothness of transition between images

delay = 30

morph(np.random.randint(0, len(dataset), num_images), num_steps, delay)
decoded_batch = model.decode(model.encode(real_batch[0]).data).data

plt.figure(figsize=(8, 8))

plt.axis("off")

plt.imshow(np.transpose(vutils.make_grid(decoded_batch, padding=2).cpu(),(1,2,0)))