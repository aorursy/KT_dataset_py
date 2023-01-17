import torch

from torch import nn



import matplotlib.pyplot as plt

import torchvision

import torchvision.transforms as transforms

import torchvision.utils as vutils



import numpy as np

from matplotlib import rcParams

rcParams["savefig.jpeg_quality"] = 80

import imageio

from pathlib import Path

import base64

from IPython import display

# we don't like warnings

# you can comment the following 2 lines if you'd like to

import warnings

warnings.filterwarnings('ignore')



# Set up a random generator seed for reproducibility

torch.manual_seed(111)



# Create a device object that points to the CPU or GPU if available

device = ""

if torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")

    



class Helper():

    @staticmethod

    def show_gif(file_path):

        """

        To show gif or video in Colab, we need to load the data and encode with base64.

        """

        with open(file_path, 'rb') as file:

            b64 = base64.b64encode(file.read()).decode('ascii')

        return display.HTML(f'<img src="data:image/gif;base64,{b64}" />')



    @staticmethod

    def make_gif(images_files, gif_name="results.gif"):

        """

        Make gif from list of images

        """

        images = [imageio.imread(file) for file in images_files]

        imageio.mimsave(gif_name, images, fps=5)

        

    @staticmethod

    def tensor_to_image(image_tensor, title, output_path, file_name, show):

        """

        Convert tensor to image to display or save to file

        """

        # Plot the image

        image = np.transpose(image_tensor,(1,2,0))

        plt.figure(figsize=(8,8))

        plt.axis("off")

        plt.title(title)

        plt.imshow(image)

        # Save to file  

        file_path = ""       

        if output_path:

            Path(output_path).mkdir(parents=True, exist_ok=True)

            file_path = f"{output_path}/{file_name}"

            plt.savefig(file_path)

        if not show: # Close the plot to not display image

            plt.close('all')

        # Return path of the saved file

        return file_path



    @staticmethod

    def show_losses(losses_generator, losses_discriminator):

        plt.figure(figsize=(10,5))

        plt.title("Generator and Discriminator Loss During Training")

        plt.plot(losses_generator,label="G")

        plt.plot(losses_discriminator,label="D")

        plt.xlabel("iterations")

        plt.ylabel("Loss")

        plt.legend()

        plt.show()
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



# Learning rate for optimizers

lr = 0.0002



# Beta1 hyperparam for Adam optimizers

beta1 = 0.5



# Initialize BCELoss function

loss_function = nn.BCELoss()



# Custom weights initialization for neural network model

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
# Load CelebA dataset from Kaggle input folder

train_set = torchvision.datasets.ImageFolder(

    root="/kaggle/input/celeba-dataset", transform=transform

)



# Create a data loader to shuffle and return data in batches for training

train_loader = torch.utils.data.DataLoader(

    train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True

)
# Get total number of batches. We print the losses after training the last batch of each epoch

num_batches = len(train_loader)



# Set how many repetitions of training with the whole dataset

num_epochs = 3



## Because the labels remain the same for every batch, we define them as constants to use for all training steps:

# Create tensor of labels for real samples with value=1 and shape is batch_size x 1

real_samples_labels = torch.ones((batch_size, 1)).to(device)

# Create tensor of labels for generated samples with value=0 and shape is batch_size x 1

generated_samples_labels = torch.zeros((batch_size, 1)).to(device) 

# Create tensor of labels for combined data

all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))



# Create batch of fixed latent vectors that we will use to visualize the progression of the generator

fixed_latent_vectors = torch.randn((batch_size, nz, 1, 1)).to(device)



# Set where to save the generated image files. We just save it temporarily

output_path = '/kaggle/temp/'
class Generator(nn.Module):

    def __init__(self, batchnorm=True):

        super().__init__()

        if batchnorm:

            self.model = nn.Sequential(

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

        else:

            self.model = nn.Sequential(

                # input is Z, going into a convolution

                nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),

                nn.ReLU(True),

                # state size. (ngf*8) x 4 x 4

                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),

                nn.ReLU(True),

                # state size. (ngf*4) x 8 x 8

                nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),

                nn.ReLU(True),

                # state size. (ngf*2) x 16 x 16

                nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),

                nn.ReLU(True),

                # state size. (ngf) x 32 x 32

                nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),

                nn.Tanh()

                # state size. (nc) x 64 x 64

            )



    def forward(self, input):

        return self.model(input)

    

    

class Discriminator(nn.Module):

    def __init__(self, batchnorm=True):

        super().__init__()

        if batchnorm:

            self.model = nn.Sequential(

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

        else:

            self.model = nn.Sequential(

                # input is (nc) x 64 x 64

                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),

                nn.LeakyReLU(0.2, inplace=True),

                # state size. (ndf) x 32 x 32

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),

                nn.LeakyReLU(0.2, inplace=True),

                # state size. (ndf*2) x 16 x 16

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),

                nn.LeakyReLU(0.2, inplace=True),

                # state size. (ndf*4) x 8 x 8

                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),

                nn.LeakyReLU(0.2, inplace=True),

                # state size. (ndf*8) x 4 x 4

                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),

                nn.Sigmoid()

            )



    def forward(self, input):

        return self.model(input)
class GAN():

    def __init__(self, batchnorm=True):

        self.batchnorm = batchnorm

        

        # Create the generator and the discriminator

        self.generator = Generator(self.batchnorm).to(device)

        self.discriminator = Discriminator(self.batchnorm).to(device)



        # Apply the weights_init function

        self.generator.apply(weights_init)

        self.discriminator.apply(weights_init)



        # Setup Adam optimizers

        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))

    

    

    def train_discriminator(self, real_samples, generated_samples, alternating=True):

        """

        Train the discriminator model by minimizing its error.

        Input: 

            real_samples: tensor of images with shape: batch_size x channel x width x height

            generated_samples: tensor of generated images with shape: batch_size x channel x width x height

            alternating: training strategy:

                alternating=True: Train the model alternately with different batches of real data and generated data.

                alternating=False: Combine real and generated data into one batch and train

        Return:

            loss_discriminator: for printing purpose

        """  

        # Clear the gradients of the discriminator to avoid accumulating them

        self.discriminator.zero_grad()

        

        if alternating: #  D(x)  ->  backward  ->  D(G(z)) -> backward

            

            # Train the discriminator with the real data

            output_discriminator_real = self.discriminator(real_samples)

            # Calculate the loss function for the discriminator to minimize its error

            loss_discriminator_real = loss_function(output_discriminator_real, real_samples_labels)

            # Calculate the gradients for the discriminator

            loss_discriminator_real.backward()



            # Train the discriminator with the generated data

            output_discriminator_generated = self.discriminator(generated_samples)

            # Calculate the loss function for the discriminator to minimize its error

            loss_discriminator_generated = loss_function(output_discriminator_generated, generated_samples_labels)

            # Calculate the gradients for the discriminator

            loss_discriminator_generated.backward()



            # Calculate the total loss of the discriminator to show later

            loss_discriminator = loss_discriminator_real + loss_discriminator_generated

            

        else: #  t = [x, G(z)]  ->  D(t)  ->  backward

            

            # Combine the real and generated data into one batch

            all_samples = torch.cat((real_samples, generated_samples))

            # Train the discriminator with the combined data

            output_discriminator = self.discriminator(all_samples)

            # Calculate the loss function for the discriminator to minimize its error

            loss_discriminator = loss_function(output_discriminator, all_samples_labels)

            # Calculate the gradients for the discriminator

            loss_discriminator.backward()

        

        # Update the weights of the discriminator

        self.optimizer_discriminator.step()



        return loss_discriminator

    

    

    def train_generator(self, output_generator):

        """

        Continue to train the generator model with its output by maximizing the discriminator error.

        Input:

            output_generator: output of the generator model when feeding the latent data

        Return:

            loss_generator: for printing purpose

        """  

        # Clear the gradients of the generator to avoid accumulating them 

        self.generator.zero_grad()

        # Get the discriminator prediction on the generator's output 

        output_discriminator_generated = self.discriminator(output_generator)

        # Calculate the loss function for the generator to maximize the discriminator error

        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)

        # Calculate the gradients for the generator 

        loss_generator.backward()

        # Update the weights of the generator

        self.optimizer_generator.step()



        return loss_generator

    



    def train(self, alternating_training_discriminator=True, output_name='results'):

        # Save the losses to visualize

        self.losses_discriminator, self.losses_generator = [], []

        self.output_files = []



        # Repeat the training process based on the number of epochs

        for epoch in range(num_epochs):

            # Load training data by batches

            for batch, (real_samples, _) in enumerate(train_loader):

                real_samples = real_samples.to(device)   

                

                ## Prepare data for training

                # Randomize tensor of latent vectors with shape (batch_size x nz x 1 x 1)

                latent_vectors = torch.randn((batch_size, nz, 1, 1)).to(device)

                # Feed the latent vectors to the generator (1 line)

                output_generator = self.generator(latent_vectors)

                # Get the generated data without its gradients (to use in training the discriminator) 

                generated_samples = output_generator.detach()



                

                ## Train the discriminator (1 line): 

                loss_discriminator = self.train_discriminator(real_samples, generated_samples, 

                                                              alternating_training_discriminator)



                ## Continue to train the generator with its output and get the loss_generator 

                loss_generator = self.train_generator(output_generator)



                self.losses_discriminator += [loss_discriminator]

                self.losses_generator += [loss_generator]



                # Print losses

                if (batch % 500 == 0) or (batch == num_batches - 1):

                    print(f"Epoch {epoch} - Batch {batch}. Loss D.: {loss_discriminator}. Loss G.: {loss_generator}")

                    title = f"After {batch} batches of {epoch} epoch(s)"

                    file_name = f"{output_name}_e{epoch:0=4d}_b{batch:0=4d}.jpg"

                    file = self.generate_images(title=title, output_path=output_path, 

                                                  file_name=file_name, show=False)

                    self.output_files.append(file)

        

        Helper.make_gif(self.output_files, output_name+'.gif')

        Helper.show_losses(self.losses_generator, self.losses_discriminator)

        return Helper.show_gif(output_name+'.gif')

        

    

    def generate_images(self, title=False, output_path=False, file_name=False, show=True):

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

            generated_samples = self.generator(fixed_latent_vectors)

            # Move the data back to the CPU and create a view of data (without gradients)

            generated_samples = generated_samples.cpu().detach()

            # Create grid of 64 generated images

            img_grid = vutils.make_grid(generated_samples[:64], padding=2, normalize=True)

            

        file_path = Helper.tensor_to_image(img_grid, title, output_path, file_name, show)

        return file_path
gan1 = GAN(batchnorm=True)

gan1.train(alternating_training_discriminator=True, output_name='gan_batchnorm_alternating')
gan2 = GAN(batchnorm=True)

gan2.train(alternating_training_discriminator=False, output_name='gan_batchnorm')
gan3 = GAN(batchnorm=False)

gan3.train(alternating_training_discriminator=True, output_name='gan_alternating')
gan4 = GAN(batchnorm=False)

gan4.train(alternating_training_discriminator=False, output_name='gan')