!pip install jovian --upgrade --quiet
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim import RMSprop
import os
from IPython.display import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt
%matplotlib inline
cifar10 = CIFAR10(root='data', 
              train=True, 
              download=True,
              transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

img, label = cifar10[0]
print('Label: ', label)
print(img[:,10:15,10:15])
torch.min(img), torch.max(img)
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)
img_norm = denorm(img)
plt.imshow(img_norm[0])
print('Label:', label)
batch_size = 64
latent_size = 100
data_loader = DataLoader(cifar10, batch_size, shuffle=True, num_workers = 2, pin_memory = True)

for img_batch, label_batch in data_loader:
    print('first batch')
    print(img_batch.shape)
    plt.imshow(img_batch[0][0], cmap='gray')
    print(label_batch)
    break
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
def wasserstein_loss(labels, output):
    return torch.mean(labels * output)
critic = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0),
            nn.Linear(1,1)
        )

# Create the critic
critic.to(device)
  
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
critic.apply(weights_init)

# Print the critic
print(critic)
G = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
G.to(device)
G.apply(weights_init)
critic_optimizer = RMSprop(critic.parameters(), lr=5e-5)
g_optimizer = RMSprop(G.parameters(), lr=5e-5)
criterion = wasserstein_loss
def reset_grad():
    critic_optimizer.zero_grad()
    g_optimizer.zero_grad()

def train_critic(images, grad_clip = 0.01):
    # Create the labels which are later used as input for the BCE loss
    real_labels = -torch.ones(batch_size, 1, 1, 1).to(device)
    fake_labels = torch.ones(batch_size, 1, 1, 1).to(device)
        
    # Loss for real images
    outputs = critic(images)
    critic_loss_real = criterion(outputs, real_labels)
    real_score = outputs

    # Loss for fake images
    z = torch.randn(batch_size, latent_size, 1, 1).to(device)
    fake_images = G(z)
    outputs = critic(fake_images)
    critic_loss_fake = criterion(outputs, fake_labels)
    fake_score = outputs

    # Combine losses
    critic_loss = critic_loss_real + critic_loss_fake
    # Reset gradients
    reset_grad()
    # Compute gradients
    critic_loss.backward()
    clip_grad_value_(critic.parameters(), grad_clip)
    # Adjust the parameters using backprop
    critic_optimizer.step()
    
    return critic_loss, real_score, fake_score
def train_generator(grad_clip = 0.01):
    # Generate fake images and calculate loss
    z = torch.randn(batch_size, latent_size, 1, 1).to(device)
    fake_images = G(z)
    labels = -torch.ones(batch_size, 1, 1, 1).to(device)
    g_loss = criterion(critic(fake_images), labels)

    # Backprop and optimize
    reset_grad()
    g_loss.backward()
    clip_grad_value_(G.parameters(), grad_clip)
    g_optimizer.step()
    return g_loss, fake_images
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
# Save some real images
for images, _ in data_loader:
    images = images.reshape(images.size(0), 3, 32, 32)
    save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'), nrow=8)
    break
   
Image(os.path.join(sample_dir, 'real_images.png'))
sample_vectors = torch.randn(batch_size, latent_size, 1, 1).to(device)

def save_fake_images(index):
    fake_images = G(sample_vectors)
    fake_images = fake_images.reshape(fake_images.size(0), 3, 32, 32)
    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    print('Saving', fake_fname)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    
# Before training
save_fake_images(0)
Image(os.path.join(sample_dir, 'fake_images-0000.png'))
%%time

num_epochs = 300
d_losses, g_losses, real_scores, fake_scores = [], [], [], []

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # Load a batch & transform to vectors
        images = images.to(device)
        if images.shape != torch.Size([64, 3, 32, 32]):
            continue
        # Train the discriminator and generator
        
        for i in range(5):    
            d_loss, real_score, fake_score = train_critic(images)
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
        print('hello')
        break
        
    # Sample and save images
    save_fake_images(epoch+1)
# Save the model checkpoints 
torch.save(G.state_dict(), 'G.ckpt')
torch.save(critic.state_dict(), 'D.ckpt')
Image('./samples/fake_images-0010.png')
Image('./samples/fake_images-0050.png')
Image('./samples/fake_images-0100.png')
Image('./samples/fake_images-0300.png')
plt.plot(d_losses, '-')
plt.plot(g_losses, '-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Discriminator', 'Generator'])
plt.title('Losses')
for i, data in enumerate(data_loader):
    print(data[0].shape)
    break
plt.plot(real_scores, '-')
plt.plot(fake_scores, '-')
plt.xlabel('epoch')
plt.ylabel('score')
plt.legend(['Real Score', 'Fake score'])
plt.title('Scores')
