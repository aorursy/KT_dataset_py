!ls ../input/celeb/processed-celeba-small/processed_celeba_small/
import shutil

shutil.copy("../input/tests/problem_unittests.py","problem_unittests.py")
# can comment out after executing

#!unzip processed_celeba_small.zip
data_dir = 'processed_celeba_small/'



"""

DON'T MODIFY ANYTHING IN THIS CELL

"""

import pickle as pkl

import matplotlib.pyplot as plt

import numpy as np

import problem_unittests as tests

#import helper



%matplotlib inline
# necessary imports

import torch

from torchvision import datasets

from torchvision import transforms
import os

import torch

from torch.utils.data import DataLoader

import torchvision

import torchvision.datasets as datasets

import torchvision.transforms as transforms
def get_dataloader(batch_size, image_size, data_dir='../input/celeb/processed-celeba-small/processed_celeba_small/'):

    """

    Batch the neural network data using DataLoader

    :param batch_size: The size of each batch; the number of images in a batch

    :param img_size: The square size of the image data (x, y)

    :param data_dir: Directory where image data is located

    :return: DataLoader with batched data

    """

    

    # TODO: Implement function and return a dataloader

    num_workers = 0

    transform = transforms.Compose([transforms.Resize(image_size), 

                                    transforms.ToTensor()])

    train_path = os.path.join(data_dir, "celeba")

    train_dataset = datasets.ImageFolder(train_path, transform)

    

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    

        

    return train_loader

# Define function hyperparameters

batch_size = 32 #changed to the batch size of 32, based on the number of images displayed in next function 

img_size = 32



"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

# Call your function and get a dataloader

celeba_train_loader = get_dataloader(batch_size, img_size)

# helper display function

def imshow(img):

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))



"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

# obtain one batch of training images

dataiter = iter(celeba_train_loader)

images, _ = dataiter.next() # _ for no labels



# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(20, 4))

plot_size=20

for idx in np.arange(plot_size):

    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])

    imshow(images[idx])
# TODO: Complete the scale function

def scale(x, feature_range=(-1, 1)):

    ''' Scale takes in an image x and returns that image, scaled

       with a feature_range of pixel values from -1 to 1. 

       This function assumes that the input x is already scaled from 0-1.'''

    # assume x is scaled to (0, 1)

    # scale to feature_range and return scaled x

    min, max = feature_range

    x = x*(max-min)+min

    return x

"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

# check scaled range

# should be close to -1 to 1

img = images[0]

scaled_img = scale(img)



print('Min: ', scaled_img.min())

print('Max: ', scaled_img.max())
import torch.nn as nn

import torch.nn.functional as F
import torch.nn as nn

import torch.nn.functional as F



# helper conv function

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):

    """Creates a convolutional layer, with optional batch normalization.

    """

    layers = []

    conv_layer = nn.Conv2d(in_channels, out_channels, 

                           kernel_size, stride, padding, bias=False)

    

    # append conv layer

    layers.append(conv_layer)



    if batch_norm:

        # append batchnorm layer

        layers.append(nn.BatchNorm2d(out_channels))

     

    # using Sequential container

    return nn.Sequential(*layers)

class Discriminator(nn.Module):



    def __init__(self, conv_dim):

        """

        Initialize the Discriminator Module

        :param conv_dim: The depth of the first convolutional layer

        """

        super(Discriminator, self).__init__()



        # complete init function

        self.conv_dim = conv_dim

        self.conv1 = conv(3,conv_dim,4,2,batch_norm=False)

        self.conv2 = conv(conv_dim,conv_dim*2,4)

        self.conv3 = conv(conv_dim*2,conv_dim*4,4)

        

        self.fc1 = nn.Linear(conv_dim*4*4*4,1)

        



    def forward(self, x):

        """

        Forward propagation of the neural network

        :param x: The input to the neural network     

        :return: Discriminator logits; the output of the neural network

        """

        # define feedforward behavior

        out = F.leaky_relu(self.conv1(x),0.2)

        out = F.leaky_relu(self.conv2(out),0.2)

        out = F.leaky_relu(self.conv3(out),0.2)

        

        out = out.view(-1,self.conv_dim*4*4*4)

        out = self.fc1(out)

        

        return out





"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

tests.test_discriminator(Discriminator)
# helper deconv function

def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):

    """Creates a transposed-convolutional layer, with optional batch normalization.

    """

    ## TODO: Complete this function

    ## create a sequence of transpose + optional batch norm layers

    layers = []

    transpose_conv_layer = nn.ConvTranspose2d(in_channels,out_channels,kernel_size,

                                             stride,padding)

    layers.append(transpose_conv_layer)

    

    if batch_norm:

        layers.append(nn.BatchNorm2d(out_channels))

        

    return nn.Sequential(*layers)
class Generator(nn.Module):

    

    def __init__(self, z_size, conv_dim):

        """

        Initialize the Generator Module

        :param z_size: The length of the input latent vector, z

        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer

        """

        super(Generator, self).__init__()



        # complete init function

        self.conv_dim = conv_dim

        self.fc = nn.Linear(z_size,conv_dim*4*4*4)

        

        #the transpose convolution layers

        self.t_conv1 = deconv(conv_dim*4,conv_dim*2,4)

        self.t_conv2 = deconv(conv_dim*2,conv_dim,4)

        self.t_conv3 = deconv(conv_dim,3,4,batch_norm = False)

        



    def forward(self, x):

        """

        Forward propagation of the neural network

        :param x: The input to the neural network     

        :return: A 32x32x3 Tensor image as output

        """

        # define feedforward behavior

        out = self.fc(x)

        #reshaping the input

        out = out.view(-1,self.conv_dim*4,4,4)

        

        out = F.relu(self.t_conv1(out))

        out = F.relu(self.t_conv2(out))

        

        out = self.t_conv3(out)

        out = F.tanh(out)

        

        return out



"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

tests.test_generator(Generator)
#imports for weight_initializations

from torch.nn import init
def weights_init_normal(m):

    """

    Applies initial weights to certain layers in a model .

    The weights are taken from a normal distribution 

    with mean = 0, std dev = 0.02.

    :param m: A module or layer in a network    

    """

    # classname will be something like:

    # `Conv`, `BatchNorm2d`, `Linear`, etc.

    classname = m.__class__.__name__

    

    # TODO: Apply initial weights to convolutional and linear layers

    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

            init.normal_(m.weight.data, 0.0, 0.02)

            

            if hasattr(m, 'bias') and m.bias is not None:

                init.constant_(m.bias.data, 0.0)

    

    
"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

def build_network(d_conv_dim, g_conv_dim, z_size):

    # define discriminator and generator

    D = Discriminator(d_conv_dim)

    G = Generator(z_size=z_size, conv_dim=g_conv_dim)



    # initialize model weights

    D.apply(weights_init_normal)

    G.apply(weights_init_normal)



    print(D)

    print()

    print(G)

    

    return D, G

# Define model hyperparams

d_conv_dim = 32

g_conv_dim = 32

z_size = 100



"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

D, G = build_network(d_conv_dim, g_conv_dim, z_size)
"""

DON'T MODIFY ANYTHING IN THIS CELL

"""

import torch



# Check for a GPU

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:

    print('No GPU found. Please use a GPU to train your neural network.')

else:

    print('Training on GPU!')
def real_loss(D_out):

    '''Calculates how close discriminator outputs are to being real.

       param, D_out: discriminator logits

       return: real loss'''

    batch_size = D_out.size(0)

    # label smoothing

#     if smooth:

#         # smooth, real labels = 0.9

#         labels = torch.ones(batch_size)*0.9

#     else:

#         labels = torch.ones(batch_size) # real labels = 1



    labels = torch.ones(batch_size)

    

    # move labels to GPU if available     

    if train_on_gpu:

        labels = labels.cuda()

    # binary cross entropy with logits loss

    criterion = nn.BCEWithLogitsLoss()

    # calculate loss

    loss = criterion(D_out.squeeze(), labels)

    return loss





def fake_loss(D_out):

    '''Calculates how close discriminator outputs are to being fake.

       param, D_out: discriminator logits

       return: fake loss'''

    batch_size = D_out.size(0)

    labels = torch.zeros(batch_size) # fake labels = 0

    if train_on_gpu:

        labels = labels.cuda()

    criterion = nn.BCEWithLogitsLoss()

    # calculate loss

    loss = criterion(D_out.squeeze(), labels)

    return loss
import torch.optim as optim



# Create optimizers for the discriminator D and generator G

lr = 0.0002

beta1 = 0.5

beta2 = 0.999



d_optimizer = optim.Adam(D.parameters(),lr,[beta1,beta2])

g_optimizer = optim.Adam(G.parameters(),lr,[beta1,beta2])
def train(D, G, n_epochs, print_every=50):

    '''Trains adversarial networks for some number of epochs

       param, D: the discriminator network

       param, G: the generator network

       param, n_epochs: number of epochs to train for

       param, print_every: when to print and record the models' losses

       return: D and G losses'''

    

    # move models to GPU

    if train_on_gpu:

        D.cuda()

        G.cuda()



    # keep track of loss and generated, "fake" samples

    samples = []

    losses = []



    # Get some fixed data for sampling. These are images that are held

    # constant throughout training, and allow us to inspect the model's performance

    sample_size=16

    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))

    fixed_z = torch.from_numpy(fixed_z).float()

    # move z to GPU if available

    if train_on_gpu:

        fixed_z = fixed_z.cuda()



    # epoch training loop

    for epoch in range(n_epochs):



        # batch training loop

        for batch_i, (real_images, _) in enumerate(celeba_train_loader):



            batch_size = real_images.size(0)

            real_images = scale(real_images)

            

            

            

            

            

            #debugging

#             print(batch_size)

#             print(real_images.size())



            # ===============================================

            #         YOUR CODE HERE: TRAIN THE NETWORKS

            # ===============================================

            

            # 1. Train the discriminator on real and fake images

            d_optimizer.zero_grad()

            

            #train with real images

            if train_on_gpu:

                real_images = real_images.cuda()

            

            D_real = D(real_images)

            d_real_loss = real_loss(D_real)

            

            #train with fake images

            

            #generate the fake images

            z = np.random.uniform(-1,1,size = (batch_size,z_size))

            z = torch.from_numpy(z).float()

            

            if train_on_gpu:

                z = z.cuda()

            

            fake_images = G(z)

            

            #now we compute the discriminator loss on fake image

            D_fake = D(fake_images)

            d_fake_loss = fake_loss(D_fake)

            

            #add up loss and perform back prop

            d_loss = d_fake_loss + d_real_loss

            d_loss.backward()

            d_optimizer.step()



            # 2. Train the generator with an adversarial loss

            g_optimizer.zero_grad()

            

            #generate the fake image

            z = np.random.uniform(-1, 1, size=(batch_size, z_size))

            z = torch.from_numpy(z).float()

            if train_on_gpu:

                z = z.cuda()

            fake_images = G(z)

            

            D_fake = D(fake_images)

            g_loss = real_loss(D_fake)

            

            #backpropagation

            g_loss.backward()

            g_optimizer.step()

            

                       

            

            # ===============================================

            #              END OF YOUR CODE

            # ===============================================



            # Print some loss stats

            if batch_i % print_every == 0:

                # append discriminator loss and generator loss

                losses.append((d_loss.item(), g_loss.item()))

                # print discriminator and generator loss

                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(

                        epoch+1, n_epochs, d_loss.item(), g_loss.item()))





        ## AFTER EACH EPOCH##    

        # this code assumes your generator is named G, feel free to change the name

        # generate and save sample, fake images

        G.eval() # for generating samples

        samples_z = G(fixed_z)

        samples.append(samples_z)

        G.train() # back to training mode



    # Save training generator samples

    with open('train_samples.pkl', 'wb') as f:

        pkl.dump(samples, f)

    

    # finally return losses

    return losses
# set number of epochs 

n_epochs = 100





"""

DON'T MODIFY ANYTHING IN THIS CELL

"""

# call training function

losses = train(D, G, n_epochs=n_epochs)
fig, ax = plt.subplots()

losses = np.array(losses)

plt.plot(losses.T[0], label='Discriminator', alpha=0.5)

plt.plot(losses.T[1], label='Generator', alpha=0.5)

plt.title("Training Losses")

plt.legend()
# helper function for viewing a list of passed in sample images

def view_samples(epoch, samples):

    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)

    for ax, img in zip(axes.flatten(), samples[epoch]):

        img = img.detach().cpu().numpy()

        img = np.transpose(img, (1, 2, 0))

        img = ((img + 1)*255 / (2)).astype(np.uint8)

        ax.xaxis.set_visible(False)

        ax.yaxis.set_visible(False)

        im = ax.imshow(img.reshape((32,32,3)))
# Load samples from generator, taken while training

with open('train_samples.pkl', 'rb') as f:

    samples = pkl.load(f)
_ = view_samples(-1, samples)