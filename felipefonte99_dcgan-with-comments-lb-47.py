import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.nn.parallel

import torch.optim as optim

import torch.utils.data

import torchvision

import torchvision.datasets as dset

import torchvision.transforms as transforms

from torchvision.utils import save_image

from torch.autograd import Variable

from scipy.stats import truncnorm

import xml.etree.ElementTree as ET

import random

import numpy as np

import pandas as pd

from tqdm import tqdm_notebook as tqdm

import os

print(os.listdir("../input"))
# seed_everything

def seed_everything(seed=42):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

seed_everything()
# Setting some hyperparameters

batchSize = 48 # We set the size of the batch

imageSize = 64 # We set the size of the generated images (64x64)

nz = 128

real_label = 0.5

fake_label = 0

slope=0.2
print('Number of images: ', len(os.listdir('../input/all-dogs/all-dogs')))
black_list=[] # images we don't use for trainning. Not used.
class FullCroppedDogsFolderDataset(torchvision.datasets.vision.VisionDataset):

    def __init__(self, root, transform=None, target_transform=None):

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.transform = transform

        self.target_transform = target_transform

        

        self.samples = self._load_subfolders_images(self.root)

        if len(self.samples) == 0:

            raise RuntimeError("Found 0 files in subfolders of: {}".format(self.root))

            

    def _load_subfolders_images(self, root):

        IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')



        def is_valid_file(x):

            return torchvision.datasets.folder.has_file_allowed_extension(x, IMG_EXTENSIONS)



        required_transforms = torchvision.transforms.Compose([

            torchvision.transforms.Resize(64),

            torchvision.transforms.CenterCrop(64),

        ])



        imgs = []



        paths = []

        for root, _, fnames in sorted(os.walk(root)):

            for fname in sorted(fnames):

                path = os.path.join(root, fname)

                paths.append(path)



        pbar = tqdm(paths, desc='Loading cropped images')



        for path in pbar:

            file = path.split('/')[-1]

            if is_valid_file(path) and file not in black_list:

                # Load image

                img = torchvision.datasets.folder.default_loader(path)



                # Get bounding boxes

                annotation_basename = os.path.splitext(os.path.basename(path))[0]

                annotation_dirname = next(dirname for dirname in os.listdir('../input/annotation/Annotation/') if dirname.startswith(annotation_basename.split('_')[0]))

                annotation_filename = os.path.join('../input/annotation/Annotation', annotation_dirname, annotation_basename)

                tree = ET.parse(annotation_filename)

                root = tree.getroot()

                objects = root.findall('object')

                max_dogs_per_image = 1

                

                if len(objects) <= max_dogs_per_image:

                    for o in objects:

                        bndbox = o.find('bndbox')

                        xmin = int(bndbox.find('xmin').text)

                        ymin = int(bndbox.find('ymin').text)

                        xmax = int(bndbox.find('xmax').text)

                        ymax = int(bndbox.find('ymax').text)



                        w = xmax - xmin

                        h = ymax - ymin

                        if h > w:

                            diff = h - w

                            xmin = xmin - diff/2

                            xmax = xmax + diff/2

                            xmax = min(xmax,img.width)

                            xmin = max(xmin,0)

                        if w > h:

                            diff = w - h

                            ymin = ymin - diff/2

                            ymax = ymax + diff/2

                            ymax = min(ymax,img.height)

                            ymin = max(ymin,0)



                        bbox = (xmin, ymin, xmax, ymax)



                        object_img = required_transforms(img.crop(bbox))

                        imgs.append(object_img)

                

                pbar.set_postfix_str("{} cropped images loaded".format(len(imgs)))



        return imgs

    

    def __getitem__(self, index):

        sample = self.samples[index]

        target = 1

        

        if self.transform is not None:

            sample = self.transform(sample)

        if self.target_transform is not None:

            target = self.target_transform(target)



        return sample, target



    def __len__(self):

        return len(self.samples)
# Creating the transformations

random_transforms = [transforms.RandomRotation(degrees=5)]

transform = transforms.Compose([transforms.Resize(imageSize), transforms.CenterCrop(imageSize), transforms.RandomHorizontalFlip(p=0.5),

                                transforms.RandomApply(random_transforms, p=0.5), transforms.ToTensor(),

                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



# Loading the dataset

load_cropped = True

if load_cropped:

    dataset = FullCroppedDogsFolderDataset(root='../input/all-dogs/', transform=transform)

else:

    dataset = dset.ImageFolder(root='../input/all-dogs/', transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 4) # We use dataLoader to get the images of the training set batch by batch.
print('Number of images: ', len(dataset))
# Visualize some dogs. The images are dark due to the normalize. At the end we will unnormalize.

n = 10

axes = plt.subplots(figsize=(4*n, 4*n), ncols=n, nrows=n)[1]

for i, ax in enumerate(axes.flatten()):

    ax.imshow(dataset[i][0].permute(1, 2, 0).detach().numpy()) # without permute the shape is (3, 64, 64). With permute (64, 64, 3)

plt.show()
# Decide which device we want to run on

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.

def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1:

        m.weight.data.normal_(1.0, 0.02)

        m.bias.data.fill_(0)
# Defining the generator

class G(nn.Module): # We introduce a class to define the generator.



    def __init__(self, nz=128, channels=3): # We introduce the __init__() function that will define the architecture of the generator.

        super(G, self).__init__() # We inherit from the nn.Module tools.

        

        self.nz = nz

        self.channels = channels

        

        self.main = nn.Sequential( # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).

            nn.ConvTranspose2d(self.nz, 1024, 4, 1, 0, bias = False), # We start with an inversed convolution.

            #Arguments: size of input of vector, number of features maps of output, size of kernel 4 by 4, stride, padding

            nn.BatchNorm2d(1024), # We normalize all the features along the dimension of the batch.

            nn.ReLU(True), # We apply a ReLU rectification to break the linearity.

            

            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias = False), # We add another inversed convolution.

            nn.BatchNorm2d(512),

            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),

            nn.BatchNorm2d(256),

            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),

            nn.BatchNorm2d(128),

            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),

            nn.BatchNorm2d(64),

            nn.ReLU(True),

            nn.ConvTranspose2d(64, self.channels, 3, 1, 1, bias = False),

            nn.Tanh() # We apply a Tanh rectification to break the linearity and stay between -1 and +1.

        )



    def forward(self, input): # We define the forward function that takes as argument an input that will be fed to the neural network with randon noise, and that will return the output containing the generated images.

        output = self.main(input) # We forward propagate the signal through the whole neural network of the generator defined by self.main.

        return output # We return the output containing the generated images.
# Creating the generator

netG = G(nz).to(device) # We create the generator object.

netG.apply(weights_init) # We initialize all the weights of its neural network.
# Defining the discriminator

class D(nn.Module): # We introduce a class to define the discriminator.



    def __init__(self, channels=3): # We introduce the __init__() function that will define the architecture of the discriminator.

        super(D, self).__init__() # We inherit from the nn.Module tools.

        

        self.channels = channels

        

        self.main = nn.Sequential( # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).

            nn.Conv2d(self.channels, 64, 4, 2, 1, bias = False), # We start with a convolution. 3 channels is the output of the generator. 64 feature maps. kernel: 4 x 4, stride: 2 and padding: 1

            nn.LeakyReLU(slope, inplace = True), # We apply a LeakyReLU.

            nn.Conv2d(64, 128, 4, 2, 1, bias = False), # We add another convolution.

            nn.BatchNorm2d(128), # We normalize all the features along the dimension of the batch.

            nn.LeakyReLU(slope, inplace = True),

            nn.Conv2d(128, 256, 4, 2, 1, bias = False),

            nn.BatchNorm2d(256),

            nn.LeakyReLU(slope, inplace = True),

            nn.Conv2d(256, 512, 4, 2, 1, bias = False),

            nn.BatchNorm2d(512),

            nn.LeakyReLU(slope, inplace = True),

            nn.Conv2d(512, 1, 4, 1, 0, bias = False), # We add another convolution. the final output is 1

            nn.Sigmoid() # We apply a Sigmoid rectification to break the linearity and stay between 0 and 1.

        )



    def forward(self, input): # We define the forward function that takes as argument an input that will be fed to the neural network, and that will return the output which will be a value between 0 and 1.

        output = self.main(input) # We forward propagate the signal through the whole neural network of the discriminator defined by self.main.

        return output.view(-1) # We return the output which will be a value between 0 and 1. view(-1) is to flatten the result
# Creating the discriminator

netD = D().to(device) # We create the discriminator object.

netD.apply(weights_init) # We initialize all the weights of its neural network.
%%time

# Training the DCGANs

n_epochs = 286

criterion = nn.BCELoss() # We create a criterion object that will measure the error between the prediction and the target. BCE means Binary Cross Entropy

optimizerD = optim.Adam(netD.parameters(), lr = 0.0003, betas = (0.5, 0.999)) # We create the optimizer object of the discriminator.

optimizerG = optim.Adam(netG.parameters(), lr = 0.0003, betas = (0.5, 0.999)) # We create the optimizer object of the generator.



lr_schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerD, factor=0.9, patience=37, min_lr=0.0002) # learning rate decay

lr_schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG, factor=0.9, patience=37, min_lr=0.0002) # learning rate decay



G_losses = [] # list to record the loss of the generator

D_losses = [] # list to record the loss of the discriminator



for epoch in tqdm(range(n_epochs)): # We iterate over the epochs.



    for i, data in enumerate(dataloader, 0): # We iterate over the images of the dataset.

        

        D_losses_epoch = 0

        G_losses_epoch = 0

        

        # 1st Step: Updating the weights of the neural network of the discriminator

        netD.zero_grad() # We initialize to 0 the gradients of the discriminator with respect to the weights.

        

        # Training the discriminator with a real image of the dataset

        input = data[0].to(device) # We get a real image of the dataset which will be used to train the discriminator. data is the mini batch, it has the images and labels

        

        batch_size = input.size(0)

        target = torch.full((batch_size, 1), real_label, device=device)

        output = netD(input) # We forward propagate this real image into the neural network of the discriminator to get the prediction (a value between 0 and 1).

        errD_real = criterion(output, target) # We compute the loss between the predictions (output) and the target.

        

        # Training the discriminator with a fake image generated by the generator

        noise = Variable(torch.randn(batch_size, nz, 1, 1, device=device)) # We make a random input vector (noise) of the generator.

        target = torch.full((batch_size, 1), fake_label, device=device)

        fake = netG(noise) # We forward propagate this random input vector into the neural network of the generator to get some fake generated images.        

        output = netD(fake.detach()) # We forward propagate the fake generated images into the neural network of the discriminator to get the prediction.

        errD_fake = criterion(output, target) # We compute the loss between the prediction (output) and the target.

        

        # Backpropagating the total error

        errD = errD_real + errD_fake # We compute the total error of the discriminator.

        errD.backward() # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the discriminator.

        optimizerD.step() # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the discriminator.

        D_losses_epoch += errD.item()

        

        # 2nd Step: Updating the weights of the neural network of the generator

        netG.zero_grad() # We initialize to 0 the gradients of the generator with respect to the weights.

        target = torch.full((batch_size, 1), real_label, device=device)

        output = netD(fake) # We forward propagate the fake generated images into the neural network of the discriminator to get the prediction (a value between 0 and 1).

        errG = criterion(output, target) # We compute the loss between the prediction (output between 0 and 1) and the target.

        errG.backward() # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the generator.

        optimizerG.step() # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the generator.

        G_losses_epoch += errG.item()

        

        # Saving losses for plotting later

        D_losses.append(errD.item())

        G_losses.append(errG.item())

        

    lr_schedulerD.step(D_losses_epoch)

    lr_schedulerG.step(G_losses_epoch)

    

    if epoch % 37 == 0: # code to print the current learning rate

        for param_group in optimizerD.param_groups:

            print('epoch', epoch, 'lr D:', param_group['lr'])

        for param_group in optimizerG.param_groups:

            print('epoch', epoch, 'lr G:', param_group['lr'])
# Plotting the losses

plt.figure(figsize=(10,5))

plt.title("Generator and Discriminator Loss During Training")

plt.plot(G_losses,label="G")

plt.plot(D_losses,label="D")

plt.xlabel("iterations")

plt.ylabel("Loss")

plt.legend()

plt.show()
def truncated_normal(size, threshold=1):

    values = truncnorm.rvs(-threshold, threshold, size=size)

    return values
# Creating 10000 images

if not os.path.exists('../output_images'):

    os.mkdir('../output_images')

im_batch_size = 50

n_images=10000



for i_batch in range(0, n_images, im_batch_size):

        

    # Images generation without truncnorm

    #noise = Variable(torch.randn(im_batch_size, nz, 1, 1, device=device))

    #gen_images = netG(noise)

    

    # Images generation with truncnorm

    z = truncated_normal((im_batch_size, nz, 1, 1), threshold=1)

    gen_z = torch.from_numpy(z).float().to(device)

    gen_images = netG(gen_z)

    

    gen_images.mul_(0.5).add_(0.5) # unnormalize

    images = gen_images.to('cpu').clone().detach()

    images = images.numpy().transpose(0, 2, 3, 1)

    

    for i_image in range(gen_images.size(0)):

        save_image(gen_images[i_image, :, :, :], os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))

# Zip folder

import shutil

shutil.make_archive('images', 'zip', '../output_images')
# Displaing 32 images

fig = plt.figure(figsize=(25, 16))

for i, j in enumerate(images[:32]):

    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])

    plt.imshow(j)