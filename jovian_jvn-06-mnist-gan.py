import torch

import torchvision

from torchvision.transforms import ToTensor, Normalize, Compose

from torchvision.datasets import MNIST



mnist = MNIST(root='data', 

              train=True, 

              download=True,

              transform=Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))
img, label = mnist[0]

print('Label: ', label)

print(img[:,10:15,10:15])

torch.min(img), torch.max(img)
def denorm(x):

    out = (x + 1) / 2

    return out.clamp(0, 1)
import matplotlib.pyplot as plt

%matplotlib inline



img_norm = denorm(img)

plt.imshow(img_norm[0], cmap='gray')

print('Label:', label)
from torch.utils.data import DataLoader



batch_size = 100

data_loader = DataLoader(mnist, batch_size, shuffle=True)
for img_batch, label_batch in data_loader:

    print('first batch')

    print(img_batch.shape)

    plt.imshow(img_batch[0][0], cmap='gray')

    print(label_batch)

    break
# Device configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
image_size = 784

hidden_size = 256
import torch.nn as nn



D = nn.Sequential(

    nn.Linear(image_size, hidden_size),

    nn.LeakyReLU(0.2),

    nn.Linear(hidden_size, hidden_size),

    nn.LeakyReLU(0.2),

    nn.Linear(hidden_size, 1),

    nn.Sigmoid())
D.to(device);
latent_size = 64
G = nn.Sequential(

    nn.Linear(latent_size, hidden_size),

    nn.ReLU(),

    nn.Linear(hidden_size, hidden_size),

    nn.ReLU(),

    nn.Linear(hidden_size, image_size),

    nn.Tanh())
y = G(torch.randn(2, latent_size))

gen_imgs = denorm(y.reshape((-1, 28,28)).detach())
plt.imshow(gen_imgs[0], cmap='gray');
plt.imshow(gen_imgs[1], cmap='gray');
G.to(device);
criterion = nn.BCELoss()

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
def reset_grad():

    d_optimizer.zero_grad()

    g_optimizer.zero_grad()



def train_discriminator(images):

    # Create the labels which are later used as input for the BCE loss

    real_labels = torch.ones(batch_size, 1).to(device)

    fake_labels = torch.zeros(batch_size, 1).to(device)

        

    # Loss for real images

    outputs = D(images)

    d_loss_real = criterion(outputs, real_labels)

    real_score = outputs



    # Loss for fake images

    z = torch.randn(batch_size, latent_size).to(device)

    fake_images = G(z)

    outputs = D(fake_images)

    d_loss_fake = criterion(outputs, fake_labels)

    fake_score = outputs



    # Combine losses

    d_loss = d_loss_real + d_loss_fake

    # Reset gradients

    reset_grad()

    # Compute gradients

    d_loss.backward()

    # Adjust the parameters using backprop

    d_optimizer.step()

    

    return d_loss, real_score, fake_score
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
def train_generator():

    # Generate fake images and calculate loss

    z = torch.randn(batch_size, latent_size).to(device)

    fake_images = G(z)

    labels = torch.ones(batch_size, 1).to(device)

    g_loss = criterion(D(fake_images), labels)



    # Backprop and optimize

    reset_grad()

    g_loss.backward()

    g_optimizer.step()

    return g_loss, fake_images
import os



sample_dir = 'samples'

if not os.path.exists(sample_dir):

    os.makedirs(sample_dir)
from IPython.display import Image

from torchvision.utils import save_image



# Save some real images

for images, _ in data_loader:

    images = images.reshape(images.size(0), 1, 28, 28)

    save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'), nrow=10)

    break

   

Image(os.path.join(sample_dir, 'real_images.png'))
sample_vectors = torch.randn(batch_size, latent_size).to(device)



def save_fake_images(index):

    fake_images = G(sample_vectors)

    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)

    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)

    print('Saving', fake_fname)

    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=10)

    

# Before training

save_fake_images(0)

Image(os.path.join(sample_dir, 'fake_images-0000.png'))
%%time



num_epochs = 300

total_step = len(data_loader)

d_losses, g_losses, real_scores, fake_scores = [], [], [], []



for epoch in range(num_epochs):

    for i, (images, _) in enumerate(data_loader):

        # Load a batch & transform to vectors

        images = images.reshape(batch_size, -1).to(device)

        

        # Train the discriminator and generator

        d_loss, real_score, fake_score = train_discriminator(images)

        g_loss, fake_images = train_generator()

        

        # Inspect the losses

        if (i+1) % 200 == 0:

            d_losses.append(d_loss.item())

            g_losses.append(g_loss.item())

            real_scores.append(real_score.mean().item())

            fake_scores.append(fake_score.mean().item())

            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 

                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 

                          real_score.mean().item(), fake_score.mean().item()))

        

    # Sample and save images

    save_fake_images(epoch+1)
# Save the model checkpoints 

torch.save(G.state_dict(), 'G.ckpt')

torch.save(D.state_dict(), 'D.ckpt')
Image('./samples/fake_images-0010.png')
Image('./samples/fake_images-0050.png')
Image('./samples/fake_images-0100.png')
Image('./samples/fake_images-0300.png')
import cv2

import os

from IPython.display import FileLink



vid_fname = 'gans_training.avi'



files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if 'fake_images' in f]

files.sort()



out = cv2.VideoWriter(vid_fname,cv2.VideoWriter_fourcc(*'MP4V'), 8, (302,302))

[out.write(cv2.imread(fname)) for fname in files]

out.release()

FileLink('gans_training.avi')
plt.plot(d_losses, '-')

plt.plot(g_losses, '-')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend(['Discriminator', 'Generator'])

plt.title('Losses');
plt.plot(real_scores, '-')

plt.plot(fake_scores, '-')

plt.xlabel('epoch')

plt.ylabel('score')

plt.legend(['Real Score', 'Fake score'])

plt.title('Scores');
# Install the library

!pip install jovian --upgrade -q
# Import the library

import jovian
jovian.commit(outputs=['samples/fake_images-0000.png', 

                       'samples/fake_images-0010.png', 

                       'samples/fake_images-0025.png', 

                       'samples/fake_images-0050.png', 

                       'samples/fake_images-0100.png', 

                       'samples/fake_images-0150.png', 

                       'samples/fake_images-0200.png', 

                       'samples/fake_images-0250.png', 

                       'samples/fake_images-0300.png', 

                       'gans_training.avi'], 

              project='06-mnist-gan')