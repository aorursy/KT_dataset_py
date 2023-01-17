project_name = 'Painting-GANs'
# Uncomment and install libraries if imports fail

# !conda install numpy pandas pytorch torchvision cpuonly -c pytorch -y

# !pip install matplotlib --upgrade --quiet
DATA_DIR = '../input/painting'
import os



print(os.listdir(DATA_DIR))
print(os.listdir(DATA_DIR+'/a2')[:10])
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder

import torchvision.transforms as T
image_size = 64

batch_size = 128

stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
train_ds = ImageFolder(DATA_DIR, transform=T.Compose([

    T.Resize(image_size),

    T.CenterCrop(image_size),

    T.ToTensor(),

    T.Normalize(*stats)]))



train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
import torch

from torchvision.utils import make_grid

import matplotlib.pyplot as plt

%matplotlib inline
def denorm(img_tensors):

    return img_tensors * stats[1][0] + stats[0][0]
def show_images(images, nmax=64):

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xticks([]); ax.set_yticks([])

    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))



def show_batch(dl, nmax=64):

    for images, _ in dl:

        show_images(images, nmax)

        break
show_batch(train_dl)
#!pip install jovian --upgrade --quiet
#import jovian
#jovian.commit(project=project_name, environment=None)
def get_default_device():

    """Pick GPU if available, else CPU"""

    if torch.cuda.is_available():

        return torch.device('cuda')

    else:

        return torch.device('cpu')

    

def to_device(data, device):

    """Move tensor(s) to chosen device"""

    if isinstance(data, (list,tuple)):

        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)



class DeviceDataLoader():

    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):

        self.dl = dl

        self.device = device

        

    def __iter__(self):

        """Yield a batch of data after moving it to device"""

        for b in self.dl: 

            yield to_device(b, self.device)



    def __len__(self):

        """Number of batches"""

        return len(self.dl)
device = get_default_device()

device
train_dl = DeviceDataLoader(train_dl, device)
import torch.nn as nn
discriminator = nn.Sequential(

    # in: 3 x 64 x 64



    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),

    nn.BatchNorm2d(64),

    nn.LeakyReLU(0.2, inplace=True),

    # out: 64 x 32 x 32



    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),

    nn.BatchNorm2d(128),

    nn.LeakyReLU(0.2, inplace=True),

    # out: 128 x 16 x 16



    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),

    nn.BatchNorm2d(256),

    nn.LeakyReLU(0.2, inplace=True),

    # out: 256 x 8 x 8



    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),

    nn.BatchNorm2d(512),

    nn.LeakyReLU(0.2, inplace=True),

    # out: 512 x 4 x 4



    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),

    # out: 1 x 1 x 1



    nn.Flatten(),

    nn.Sigmoid())
discriminator = to_device(discriminator, device)
latent_size = 128
generator = nn.Sequential(

    # in: latent_size x 1 x 1



    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),

    nn.BatchNorm2d(512),

    nn.ReLU(True),

    # out: 512 x 4 x 4



    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),

    nn.BatchNorm2d(256),

    nn.ReLU(True),

    # out: 256 x 8 x 8



    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),

    nn.BatchNorm2d(128),

    nn.ReLU(True),

    # out: 128 x 16 x 16



    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),

    nn.BatchNorm2d(64),

    nn.ReLU(True),

    # out: 64 x 32 x 32



    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),

    nn.Tanh()

    # out: 3 x 64 x 64

)
xb = torch.randn(batch_size, latent_size, 1, 1) # random latent tensors

fake_images = generator(xb)

print(fake_images.shape)

show_images(fake_images)
generator = to_device(generator, device)
def train_discriminator(real_images, opt_d):

    # Clear discriminator gradients

    opt_d.zero_grad()



    # Pass real images through discriminator

    real_preds = discriminator(real_images)

    real_targets = torch.ones(real_images.size(0), 1, device=device)

    real_loss = F.binary_cross_entropy(real_preds, real_targets)

    real_score = torch.mean(real_preds).item()

    

    # Generate fake images

    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)

    fake_images = generator(latent)



    # Pass fake images through discriminator

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
def save_samples(index, latent_tensors, show=True):

    fake_images = generator(latent_tensors)

    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)

    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)

    print('Saving', fake_fname)

    if show:

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.set_xticks([]); ax.set_yticks([])

        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
save_samples(0, fixed_latent)
#jovian.commit(project=project_name, environment=None)
from tqdm.notebook import tqdm

import torch.nn.functional as F
def fit(epochs, lr, start_idx=1):

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

        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(

            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))

    

        # Save generated images

        save_samples(epoch+start_idx, fixed_latent, show=False)

    

    return losses_g, losses_d, real_scores, fake_scores
lr = 0.00002

epochs = 1000
#jovian.reset()

#jovian.log_hyperparams(lr=lr, epochs=epochs)
history = fit(epochs, lr)
losses_g, losses_d, real_scores, fake_scores = history
#jovian.log_metrics(loss_g=losses_g[-1], 

 #                  loss_d=losses_d[-1], 

  #                 real_score=real_scores[-1], 

   #                fake_score=fake_scores[-1])
# Save the model checkpoints 

torch.save(generator.state_dict(), 'G.ckpt')

torch.save(discriminator.state_dict(), 'D.ckpt')
from IPython.display import Image
Image('./generated/generated-images-0001.png')
Image('./generated/generated-images-0005.png')
Image('./generated/generated-images-0025.png')
Image('./generated/generated-images-0050.png')
Image('./generated/generated-images-0100.png')
Image('./generated/generated-images-0500.png')
import cv2

import os



vid_fname = 'painting_gans_training.avi'



files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if 'generated' in f]

files.sort()



out = cv2.VideoWriter(vid_fname,cv2.VideoWriter_fourcc(*'MP4V'), 1, (530,530))

[out.write(cv2.imread(fname)) for fname in files]

out.release()
plt.plot(losses_d, '-')

plt.plot(losses_g, '-')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend(['Discriminator', 'Generator'])

plt.title('Losses');
plt.plot(real_scores, '-')

plt.plot(fake_scores, '-')

plt.xlabel('epoch')

plt.ylabel('score')

plt.legend(['Real', 'Fake'])

plt.title('Scores');
#jovian.commit(project=project_name, 

 #             outputs=['G.ckpt', 'D.ckpt', 'painting_gans_training.avi'], 

  #            environment=None)