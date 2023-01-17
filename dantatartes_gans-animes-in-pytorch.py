import torch

import torch.nn as nn

from torchvision import transforms, datasets

import torchvision.utils as vutils

from torch.autograd import Variable



import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm
num_channels = 3

latent_size = 100

base_size, image_size, batch_size = 64, 64, 64

torch.cuda.set_device("cuda:0")
transform = transforms.Compose([

    transforms.Resize(image_size),

    transforms.CenterCrop(image_size),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # makes value in between [-1, 1]

])



dataset = datasets.ImageFolder('/kaggle/input/', transform=transform)

loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
real_batch = next(iter(loader))

plt.figure(figsize=(8, 8))

plt.axis("off")

plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:32], padding=2, normalize=True).cpu(),(1,2,0)))
class Flatten(nn.Module):

    def forward(self, input):

        return input.view(input.size(0), -1)



    

class UnFlatten(nn.Module):

    def forward(self, input):

        return input.view(input.size(0), 128, image_size//4, image_size//4)



def discriminator_block(in_filters, out_filters, bn=True):

            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),

                     nn.LeakyReLU(0.2, inplace=True),

                     nn.Dropout2d(0.25)

                    ]

            if bn:

                block.append(nn.BatchNorm2d(out_filters, 0.8))

            return block
G = nn.Sequential(

    nn.Linear(100, 128 * (image_size//4) ** 2),

    UnFlatten(),

    nn.BatchNorm2d(128),

    nn.Upsample(scale_factor=2),

    nn.Conv2d(128, 128, 3, stride=1, padding=1),

    nn.BatchNorm2d(128, 0.8),

    nn.LeakyReLU(0.2, inplace=True),

    nn.Upsample(scale_factor=2),

    nn.Conv2d(128, 64, 3, stride=1, padding=1),

    nn.BatchNorm2d(64, 0.8),

    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(64, num_channels, 3, stride=1, padding=1),

    nn.Tanh(),

)
D = nn.Sequential(

    *discriminator_block(num_channels, 16, bn=False),

    *discriminator_block(16, 32),

    *discriminator_block(32, 64),

    *discriminator_block(64, 128),

    Flatten(),

    nn.Linear(128 * (image_size//2**4) ** 2, 1), 

    nn.Sigmoid()

)
def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:

        nn.init.normal_(m.weight.data, 1.0, 0.02)

        nn.init.constant_(m.bias.data, 0)





# recursively apply weights initialization on every submodule

G.apply(weights_init)

D.apply(weights_init)
def draw_my_picture():

    Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_size))))

    img = G(z).cpu().data[0]

    img = img.view((num_channels, image_size, image_size)).transpose(0, 1).transpose(1, 2).cpu().numpy()

    plt.axis('off')

    plt.imshow(img.reshape(image_size, image_size, num_channels))

    plt.show()
try:

    D.load_state_dict(torch.load('D.pth'))

    G.load_state_dict(torch.load('G.pth'))

except:

    print("Weights not found ):")
cuda = True if torch.cuda.is_available() else False

adversarial_loss = torch.nn.BCELoss()

if cuda:

    G.cuda()

    D.cuda()

    adversarial_loss.cuda()

    Tensor = torch.cuda.FloatTensor

else:

    Tensor = torch.FloatTensor
num_epochs = 30

learning_rate = 1e-3



optim_G = torch.optim.Adam(G.parameters(), lr=learning_rate)

optim_D = torch.optim.Adam(D.parameters(), lr=learning_rate)

criterion_G = nn.BCELoss()

criterion_D = nn.BCELoss()

for epoch in tqdm(range(num_epochs)):

    for imgs, _ in loader:

        # Train Generator

        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)

        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)



        real_imgs = Variable(imgs.type(Tensor))



        optim_G.zero_grad()



        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_size)))) # Sample noise

        gen_imgs = G(z)

        G_loss = adversarial_loss(D(gen_imgs), valid)



        G_loss.backward()

        optim_G.step()



        #  Train Discriminator

        optim_D.zero_grad()

        real_loss = adversarial_loss(D(real_imgs), valid)

        fake_loss = adversarial_loss(D(gen_imgs.detach()), fake)

        D_loss = (real_loss + fake_loss) / 2



        D_loss.backward()

        optim_D.step()

    if epoch % 2 == 0:

        D.eval()

        G.eval()

        draw_my_picture()        

        print(f"D_loss: {D_loss.item():.4f} G_loss: {G_loss.item():.4f}")

        torch.save(D.state_dict(), 'D.pth')

        torch.save(G.state_dict(), 'G.pth')

        D.train()

        G.train()
z = Variable(Tensor(np.random.normal(0, 1, (32, latent_size)))) # Sample noise

gen_imgs = G(z).detach().cpu()

plt.figure(figsize=(8, 8))

plt.axis("off")

plt.imshow(np.transpose(vutils.make_grid(gen_imgs, padding=2, normalize=True).cpu(),(1,2,0)))