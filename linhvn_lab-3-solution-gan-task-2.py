import torch

from torch import nn



import matplotlib.pyplot as plt

import torchvision

import torchvision.transforms as transforms

import torchvision.utils as vutils



import numpy as np

from matplotlib import rcParams

rcParams["savefig.jpeg_quality"] = 50

import imageio

from pathlib import Path

# we don't like warnings

# you can comment the following 2 lines if you'd like to

import warnings

warnings.filterwarnings('ignore')



# Set up a random generator seed so that the experiment can be replicated identically on any machine

torch.manual_seed(111)



# Create a device object that points to the CPU or GPU if available

device = ""

if torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")



# Batch size during training

batch_size = 128



# Set image size for the transformer

image_size = 64



# Number of channels in images. For color images this is 3

nc = 3



# Size of z latent vector (i.e. size of generator input)

nz = 100



# Size of feature maps in generator

ngf = 64



# Size of feature maps in discriminator

ndf = 64



# Define the transform to load images from CelebA dataset

transform = transforms.Compose([transforms.Resize(image_size),

                               transforms.CenterCrop(image_size),

                               transforms.ToTensor(),

                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                              ])



# Load CelebA dataset from Kaggle input folder

train_set = torchvision.datasets.ImageFolder(

    root="/kaggle/input/celeba-dataset", transform=transform

)



# Create a data loader to shuffle the data from train_set and return data in batches for training

train_loader = torch.utils.data.DataLoader(

    train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True

)
real_samples, celebA_labels = next(iter(train_loader))

plt.figure(figsize=(8,8))

plt.axis("off")

plt.title("Training Images")

plt.imshow(np.transpose(vutils.make_grid(real_samples.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
# custom weights initialization called on generator and discriminator

def weights_init(m):

    """

    Randomly initialize all weights to mean=0, stdev=0.2

    """

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:

        nn.init.normal_(m.weight.data, 1.0, 0.02)

        nn.init.constant_(m.bias.data, 0)
# Generator Code



class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()

        self.main = nn.Sequential(

            # input is Z, going into a convolution

            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),

            nn.BatchNorm2d(ngf * 8),

            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ngf * 4),

            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8

            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ngf * 2),

            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16

            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ngf),

            nn.ReLU(True),

            # state size. (ngf) x 32 x 32

            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),

            nn.Tanh()

            # state size. (nc) x 64 x 64

        )



    def forward(self, input):

        return self.main(input)
# Create the generator

generator = Generator().to(device)



# Apply the weights_init function

generator.apply(weights_init)



# Print the model

print(generator)
class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.main = nn.Sequential(

            # input is (nc) x 64 x 64

            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),

            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ndf * 2),

            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ndf * 4),

            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ndf * 8),

            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),

            nn.Sigmoid()

        )



    def forward(self, input):

        return self.main(input)
# Create the Discriminator

discriminator = Discriminator().to(device)



# Apply the weights_init function

discriminator.apply(weights_init)



# Print the model

print(discriminator)
# Number of training epochs

num_epochs = 5



# Learning rate for optimizers

lr = 0.0002



# Beta1 hyperparam for Adam optimizers

beta1 = 0.5



# Initialize BCELoss function

loss_function = nn.BCELoss()



# Setup Adam optimizers for both generator and discriminator

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
import base64

from IPython import display



def show_gif(file_path):

    """

    To show gif or video in Colab, we need to load the data and encode with base64.

    """

    with open(file_path, 'rb') as file:

        b64 = base64.b64encode(file.read()).decode('ascii')

    return display.HTML(f'<img src="data:image/gif;base64,{b64}" />')



# Create batch of fixed latent vectors that we will use to visualize the progression of the generator

fixed_latent_vectors = torch.randn((batch_size, nz, 1, 1)).to(device)



def generate_images(title=False, output_path=False, file_name=False, show=True):

    """

    Generate images from a random vector using the generator.

    Input:

    title: title of the image showing how many epochs that the generator is trained

    output_path: if you want to save file, define the output folder 

    show: display the plot or not. Set to False if you just want to save the image

    Output:

    file_path: path of the generated image file

    """     

    with torch.no_grad():

        # Generate data from fixed_latent_vectors with the generator 

        generated_samples = generator(fixed_latent_vectors)

        # Move the data back to the CPU and create a view of data (without gradients)

        generated_samples = generated_samples.cpu().detach()

    # Plot the data

    plt.figure(figsize=(8,8))

    plt.axis("off")

    plt.title(title)

    plt.imshow(np.transpose(vutils.make_grid(generated_samples.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    # Save to file  

    file_path = ""       

    if output_path:

        Path(output_path).mkdir(parents=True, exist_ok=True)

        file_path = f"{output_path}/{file_name}"

        plt.savefig(file_path)

    # Close the plot if not show

    if not show:

        plt.close('all')

    # Return path of the generated image file

    return file_path
generate_images(title='Before training')
# Get total number of batches. We print the losses after training the last batch of each epoch

num_batches = len(train_loader)



## YOUR CODE HERE ##

# Set how many repetitions of training with the whole dataset (1 line)

num_epochs = 5



# Because the labels remain the same for every batch, we create it here:

# Create tensor of labels for real samples with value=1 and shape is batch_size x 1 (1 line)

real_samples_labels = torch.ones((batch_size, 1)).to(device=device)

# Create tensor of labels for generated samples with value=0 and shape is batch_size x 1 (1 line)

generated_samples_labels = torch.zeros((batch_size, 1)).to(device=device) 

## END CODE HERE ##



# Set where to save the generated image files. We just save it temporarily

output_path = '/kaggle/temp/'

output_files = []



# Save the losses to visualize

losses_discriminator, losses_generator = [], []
def train_discriminator(real_samples, generated_samples):

    """

    Train the discriminator model by minimizing its error.

    Input: 

        real_samples: tensor of images with shape: batch_size x channel x width x height

        generated_samples: tensor of generated images with shape: batch_size x channel x width x height

    Return:

        loss_discriminator: for printing purpose

    """  

    # Clear the gradients of the discriminator to avoid accumulating them

    discriminator.zero_grad()

    

    # Train the discriminator with the real data

    output_discriminator_real = discriminator(real_samples)

    # Calculate the loss function for the discriminator to minimize its error

    loss_discriminator_real = loss_function(output_discriminator_real, real_samples_labels)

    # Calculate the gradients for the discriminator

    loss_discriminator_real.backward()



    # Train the discriminator with the generated data

    output_discriminator_generated = discriminator(generated_samples)

    # Calculate the loss function for the discriminator to minimize its error

    loss_discriminator_generated = loss_function(output_discriminator_generated, generated_samples_labels)

    # Calculate the gradients for the discriminator

    loss_discriminator_generated.backward()



    # Calculate the total loss of the discriminator to show later

    loss_discriminator = loss_discriminator_real + loss_discriminator_generated

    # Update the weights of the discriminator

    optimizer_discriminator.step()



    return loss_discriminator
def train_generator(output_generator):

    """

    Continue to train the generator model with its output by maximizing the discriminator error.

    Input:

        output_generator: output of the generator model when feeding the latent data

    Return:

        loss_generator: for printing purpose

    """  

    # Clear the gradients of the generator to avoid accumulating them (1 line)

    generator.zero_grad()

    # Get the discriminator prediction on the generator's output (1 line)

    output_discriminator_generated = discriminator(output_generator)

    # Calculate the loss function for the generator to maximize the discriminator error (1 line)

    loss_generator = loss_function(output_discriminator_generated, real_samples_labels)

    # Calculate the gradients for the generator (1 line)

    loss_generator.backward()

    # Update the weights of the generator (1 line)

    optimizer_generator.step()

    

    return loss_generator
# Repeat the training process based on the number of epochs

for epoch in range(num_epochs):

    # Load training data by batches

    for batch, (real_samples, _) in enumerate(train_loader):

        

        ## YOUR CODE HERE ##

        

        ## Prepare data for training

        # Send real samples from data loader to GPU if available (1 line)

        real_samples = real_samples.to(device=device)   

        # Randomize tensor of latent vectors with shape (batch_size x nz x 1 x 1) and send to GPU if available (1 line)

        latent_vectors = torch.randn((batch_size, nz, 1, 1)).to(device)

        # Feed the latent vectors to the generator (1 line)

        output_generator = generator(latent_vectors)

        # Create new tensor of generated data (without keeping track of the gradients of the generator, to use in training the discriminator) (1 line) 

        generated_samples = output_generator.detach()



        ## Train the discriminator with real data and generated data and get the loss_discriminator (1 line)

        loss_discriminator = train_discriminator(real_samples, generated_samples)



        ## Continue to train the generator with its output and get the loss_generator (1 line)

        loss_generator = train_generator(output_generator)



        ## END YOUR CODE HERE ##

        

        losses_discriminator += [loss_discriminator]

        losses_generator += [loss_generator]



        # Print losses

        if (batch % 500 == 0) or (batch == num_batches - 1):

            print(f"Epoch {epoch} - Batch {batch}. Loss D.: {loss_discriminator}. Loss G.: {loss_generator}")

            title = f"After {batch} batches of {epoch} epoch(s)"

            file_name = f"e{epoch:0=4d}b{batch:0=4d}.jpg"

            output_files += [generate_images(title, output_path=output_path, file_name=file_name, show=False)]
plt.figure(figsize=(10,5))

plt.title("Generator and Discriminator Loss During Training")

plt.plot(losses_generator,label="G")

plt.plot(losses_discriminator,label="D")

plt.xlabel("iterations")

plt.ylabel("Loss")

plt.legend()

plt.show()
# Make gif from list of images

images = [imageio.imread(file) for file in output_files]

imageio.mimsave('results.gif', images, fps=5)

show_gif('results.gif')
generate_images(title='After training')