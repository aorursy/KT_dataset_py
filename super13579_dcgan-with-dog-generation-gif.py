import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
import matplotlib.pyplot as plt

import numpy as np

import torch

from torch import nn, optim

import torch.nn.functional as F

from torchvision import datasets, transforms

from torchvision.utils import save_image

from torch.autograd import Variable

import os

from tqdm import tqdm
device = torch.device("cuda")
## Hyper paramatric

latent_dim = 100

lr = 0.002

img_size = 64

batch_size = 32

channels = 3

epochs = 220



#Crop 64x64 image

transform = transforms.Compose([transforms.Resize(img_size),

                                transforms.CenterCrop(img_size),

                                transforms.ToTensor(),

                                transforms.Normalize([0.5]*3,[0.5]*3)])

# Dataloader

train_data = datasets.ImageFolder('../input/all-dogs/', transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,

                                           batch_size=batch_size)



128 * img_size ** 2
## Generator

class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()



        self.init_size = img_size // 4

        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))



        self.conv_blocks = nn.Sequential(

            nn.BatchNorm2d(128),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),

            nn.BatchNorm2d(128, 0.8),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 64, 3, stride=1, padding=1),

            nn.BatchNorm2d(64, 0.8),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, channels, 3, stride=1, padding=1),

            nn.Tanh(),

        )



    def forward(self, z):

        out = self.l1(z)

        out = out.view(out.shape[0], 128, self.init_size, self.init_size)

        img = self.conv_blocks(out)

        return img
class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()



        def discriminator_block(in_filters, out_filters, bn=True):

            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]

            if bn:

                block.append(nn.BatchNorm2d(out_filters, 0.8))

            return block



        self.model = nn.Sequential(

            *discriminator_block(channels, 16, bn=False),

            *discriminator_block(16, 32),

            *discriminator_block(32, 64),

            *discriminator_block(64, 128),

        )



        # The height and width of downsampled image

        ds_size = img_size // 2 ** 4

        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2 , 1), nn.Sigmoid())



    def forward(self, img):

        #print (img.shape)

        out = self.model(img)

        out = out.view(out.shape[0], -1)

        #out = nn.Sigmoid()(out)

        #print (out.shape)

        validity = self.adv_layer(out)



        return validity
def weights_init(m):

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:

        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm2d") != -1:

        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)

        torch.nn.init.constant_(m.bias.data, 0.0)
# Initialize generator and discriminator

generator = Generator().cuda()

discriminator = Discriminator().cuda()



# weight initial

generator.apply(weights_init)

discriminator.apply(weights_init)



# Loss function

adversarial_loss = nn.BCELoss()



# Optimizers

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
def generate_image(z):

    #z = Variable(torch.cuda.FloatTensor((np.random.normal(0, 1, (1, latent_dim)))))

    #z = torch.randn(im_batch_size, latent_dim, device=device)

    gen_images = generator(z)

    images = gen_images.to("cpu").clone().detach()

    images = images.numpy().transpose(0, 2, 3, 1)

    return images
ims_animation = []

sample_interval_check = 300

valid_z = Variable(torch.cuda.FloatTensor((np.random.normal(0, 1, (1, latent_dim)))))



for epoch in range(epochs):

    d_loss_avg = 0.

    g_loss_avg = 0.

    for i, (imgs, _) in enumerate(train_loader):

        

        # Adversarial ground truths

        valid = Variable(torch.cuda.FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False).cuda()

        fake = Variable(torch.cuda.FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False).cuda()

        

        real_imgs = Variable(imgs.type(torch.cuda.FloatTensor))



        #  Train Generator

        optimizer_G.zero_grad()

        

        # Sample noise as generator input

        z = Variable(torch.cuda.FloatTensor((np.random.normal(0, 1, (imgs.shape[0], latent_dim)))))



        # Generate a batch of images

        gen_imgs = generator(z)



        # Loss measures generator's ability to fool the discriminator

        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()

        optimizer_G.step()

        

        #-------------------------------------------------------------

        #  Train Discriminator

        optimizer_D.zero_grad()



        # Measure discriminator's ability to classify real from generated samples

        real_loss = adversarial_loss(discriminator(real_imgs), valid)

        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

        d_loss = (real_loss + fake_loss) 



        d_loss.backward()

        optimizer_D.step()

        #---------------------------------------------------------------

        #  Save loss

        d_loss_avg += d_loss/len(train_loader)

        g_loss_avg += g_loss/len(train_loader)



        batches_done = epoch * len(train_loader) + i

        if i % sample_interval_check == 0:  

            ims_animation.append(generate_image(valid_z))

            

    print(

        "Epoch %d/%d [D loss: %f] [G loss: %f]"

        % (epoch, epochs, d_loss_avg.item(), g_loss_avg.item())

        )
import matplotlib.animation as animation

#from matplotlib.animation import FuncAnimation



fig = plt.figure() 



ims = []

#fig, ax = plt.subplots()

#xdata, ydata = [], []

#ln, = plt.plot([], [], 'ro',animated=True)

for j in range(len(ims_animation)):

    im = plt.imshow(ims_animation[j][0],animated=True)

    ims.append([im])

    

anim  = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000,repeat = True)



anim.save('generate_dog.gif',writer='ffmpeg')

#print ('[[file:/aaa.gif]]')

#plt.show()
if not os.path.exists('../output_images'):

    os.mkdir('../output_images')

im_batch_size = 50

n_images=10000

for i_batch in range(0, n_images, im_batch_size):

    z = Variable(torch.cuda.FloatTensor((np.random.normal(0, 1, (im_batch_size, latent_dim)))))

    #z = torch.randn(im_batch_size, latent_dim, device=device)

    gen_images = generator(z)

    images = gen_images.to("cpu").clone().detach()

    images = images.numpy().transpose(0, 2, 3, 1)

    for i_image in range(gen_images.size(0)):

        save_image(gen_images[i_image, :, :, :], os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))





import shutil

shutil.make_archive('images', 'zip', '../output_images')
for i in range(10):

    plt.imshow(images[i])

    plt.show()