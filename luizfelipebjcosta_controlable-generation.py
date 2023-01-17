import torch

from torch import nn

from tqdm.auto import tqdm

from torchvision import transforms

from torchvision.utils import make_grid

from torchvision.datasets import CelebA

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt



#definicao de variaveis

device = 'cuda'#o dispositivo a ser utilizado

#as variaveis abaixo nao devem ser mudadas por uma questao de compatibilidade com a GAN treinada

z_dim = 64

batch_size = 128



#funcao de vizualizacao

def show_tensor_images(image_tensor, num_images=16, size=(3, 64, 64), nrow=3):

    '''

    Function for visualizing images: Given a tensor of images, number of images, and

    size per image, plots and prints the images in an uniform grid.

    '''

    image_tensor = (image_tensor + 1) / 2

    image_unflat = image_tensor.detach().cpu()

    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)

    plt.imshow(image_grid.permute(1, 2, 0).squeeze())

    plt.show()
##Definicao do Gerador

class Generator(nn.Module):

    def __init__(self, z_dim=10, im_chan=3, hidden_dim=64):

        super(Generator, self).__init__()

        self.z_dim = z_dim

        # Build the neural network

        self.gen = nn.Sequential(

            self.make_gen_block(z_dim, hidden_dim * 8),

            self.make_gen_block(hidden_dim * 8, hidden_dim * 4),

            self.make_gen_block(hidden_dim * 4, hidden_dim * 2),

            self.make_gen_block(hidden_dim * 2, hidden_dim),

            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),

        )

        

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):

        '''

        Function to return a sequence of operations corresponding to a generator block of DCGAN;

        a transposed convolution, a batchnorm (except in the final layer), and an activation.

        Parameters:

            input_channels: how many channels the input feature representation has

            output_channels: how many channels the output feature representation should have

            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)

            stride: the stride of the convolution

            final_layer: a boolean, true if it is the final layer and false otherwise 

                      (affects activation and batchnorm)

        '''

        if not final_layer:

            return nn.Sequential(

                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),

                nn.BatchNorm2d(output_channels),

                nn.ReLU(inplace=True),

            )

        else:

            return nn.Sequential(

                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),

                nn.Tanh(),

            )



    def forward(self, noise):

        '''

        Function for completing a forward pass of the generator: Given a noise tensor,

        returns generated images.

        Parameters:

            noise: a noise tensor with dimensions (n_samples, z_dim)

        '''

        x = noise.view(len(noise), self.z_dim, 1, 1)

        return self.gen(x)



def get_noise(n_samples,z_dim,device='cpu'):

  return torch.randn((n_samples,z_dim),device=device)

  

##Definicao do Critico

class Critic(nn.Module):

    def __init__(self, im_chan=1, hidden_dim=64):

        super(Critic, self).__init__()

        self.crit = nn.Sequential(

            self.make_crit_block(im_chan, hidden_dim),

            self.make_crit_block(hidden_dim, hidden_dim * 2),

            self.make_crit_block(hidden_dim * 2, 1, final_layer=True),

        )



    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):

        '''

        Function to return a sequence of operations corresponding to a critic block of DCGAN;

        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).

        Parameters:

            input_channels: how many channels the input feature representation has

            output_channels: how many channels the output feature representation should have

            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)

            stride: the stride of the convolution

            final_layer: a boolean, true if it is the final layer and false otherwise 

                      (affects activation and batchnorm)

        '''

        if not final_layer:

            return nn.Sequential(

                nn.Conv2d(input_channels, output_channels, kernel_size, stride),

                nn.BatchNorm2d(output_channels),

                nn.LeakyReLU(0.2, inplace=True),

            )

        else:

            return nn.Sequential(

                nn.Conv2d(input_channels, output_channels, kernel_size, stride),

            )



    def forward(self, image):

        '''

        Function for completing a forward pass of the critic: Given an image tensor, 

        returns a 1-dimension tensor representing fake/real.

        Parameters:

            image: a flattened image tensor with dimension (im_chan)

        '''

        crit_pred = self.crit(image)

        return crit_pred.view(len(crit_pred), -1)
gen = torch.load('/kaggle/input/trained-gan-celeb-dataset/gen.pt')