#link do github: https://github.com/LFBJC/GAN

#codigo baseado no primeiro curso da serie Generative Adversarial Networks (GANs) de deeplearning.ai no coursera

import torch

from torch import nn

from PIL import Image

import matplotlib.pyplot as plt

from torchvision.datasets import CelebA # Training dataset

from torch.utils.data import DataLoader

from torchvision.utils import make_grid

from torchvision import transforms

from tqdm.auto import tqdm

import os

import copy

import random

from random import shuffle



#DEFINICAO DE VARIAVEIS

device = 'cuda' #dispositivo no qual o programa sera executado (cuda para rodar em GPU ou cpu para rodar no processador)



#definicao da GAN

z_dim = 64 #dimensao do vetor de entrada do gerador

hidden_dim = 64 #numero de canais da camada escondida

c_lambda = 10 #usado na funcao de perda do critico



#treinamento

crit_repeats = 5 #numero de vezes que o critico repete por epoca antes do gerador treinar

n_epochs = 100 #numero de epocas

batch_size = 128 #tamanho do batch

display_step = 50 #numero de passos ate a exibicao



#otimizadores

lr = 0.0002

beta_1 = 0.5

beta_2 = 0.999



# variaveis relacionadas à base de dados

image_size=64

im_chan=3



#leitura da base de dados

#transformacao a ser aplicada na base de dados

transform = transforms.Compose([

        transforms.Resize(image_size),

        transforms.CenterCrop(image_size),

        transforms.ToTensor(),

        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ])

img_list = [os.path.join(dirname, filename) for dirname,_,filenames in os.walk('/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/') for filename in filenames]

random.shuffle(img_list)

img_list=img_list[:30000]

def image_iterator(transform,img_list):

    return [transform(Image.open(p)) for p in img_list]

batches = DataLoader(

        image_iterator(transform,img_list),

        batch_size=batch_size,

        shuffle=False)
def show_tensor_images(image_tensor, num_images=25, size=(1, 64, 64)):

    '''

    Function for visualizing images: Given a tensor of images, number of images, and

    size per image, plots and prints the images in an uniform grid.

    '''

    image_tensor = (image_tensor + 1) / 2

    image_unflat = image_tensor.detach().cpu()

    image_grid = make_grid(image_unflat[:num_images], nrow=5)

    plt.imshow(image_grid.permute(1, 2, 0).squeeze())

    plt.show()

#comentei a funcao abaixo porque nao vi onde ela era utilizada, mas esta no codigo do curso,

#deve ser usada numa parte mais pra frente, mas eu vou descobrir

'''

def make_grad_hook():

    #Function to keep track of gradients for visualization purposes, 

    #which fills the grads list when using model.apply(grad_hook).

    grads = []

    def grad_hook(m):

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):

            grads.append(m.weight.grad)

    return grads, grad_hook'''
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
gen = Generator(z_dim,im_chan,hidden_dim).to(device)

gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))

crit = Critic(im_chan,hidden_dim).to(device)

crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))



def weights_init(m):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):

        torch.nn.init.normal_(m.weight, 0.0, 0.02)

    if isinstance(m, nn.BatchNorm2d):

        torch.nn.init.normal_(m.weight, 0.0, 0.02)

        torch.nn.init.constant_(m.bias, 0)

gen = gen.apply(weights_init)

crit = crit.apply(weights_init)
def get_gradient(crit, real, fake, epsilon):

    '''

    Return the gradient of the critic's scores with respect to mixes of real and fake images.

    Parameters:

        crit: the critic model

        real: a batch of real images

        fake: a batch of fake images

        epsilon: a vector of the uniformly random proportions of real/fake per mixed image

    Returns:

        gradient: the gradient of the critic's scores, with respect to the mixed image

    '''

    # Mix the images together

    mixed_images = real * epsilon + fake * (1 - epsilon)



    # Calculate the critic's scores on the mixed images

    mixed_scores = crit(mixed_images)

    

    # Take the gradient of the scores with respect to the images

    gradient = torch.autograd.grad(

        inputs=mixed_images,

        outputs=mixed_scores,

        # These other parameters have to do with the pytorch autograd engine works

        grad_outputs=torch.ones_like(mixed_scores), 

        create_graph=True,

        retain_graph=True,

    )[0]

    return gradient

def gradient_penalty(gradient):

    '''

    Return the gradient penalty, given a gradient.

    Given a batch of image gradients, you calculate the magnitude of each image's gradient

    and penalize the mean quadratic distance of each magnitude to 1.

    Parameters:

        gradient: the gradient of the critic's scores, with respect to the mixed image

    Returns:

        penalty: the gradient penalty

    '''

    # Flatten the gradients so that each row captures one image

    gradient = gradient.view(len(gradient), -1)



    # Calculate the magnitude of every row

    gradient_norm = gradient.norm(2, dim=1)

    

    # Penalize the mean squared distance of the gradient norms from 1

    penalty = torch.mean((gradient_norm-torch.ones_like(gradient_norm))**2)

    return penalty
def get_gen_loss(crit_fake_pred):

    '''

    Return the loss of a generator given the critic's scores of the generator's fake images.

    Parameters:

        crit_fake_pred: the critic's scores of the fake images

    Returns:

        gen_loss: a scalar loss value for the current batch of the generator

    '''

    return -torch.mean(crit_fake_pred)

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):

    '''

    Return the loss of a critic given the critic's scores for fake and real images,

    the gradient penalty, and gradient penalty weight.

    Parameters:

        crit_fake_pred: the critic's scores of the fake images

        crit_real_pred: the critic's scores of the real images

        gp: the unweighted gradient penalty

        c_lambda: the current weight of the gradient penalty 

    Returns:

        crit_loss: a scalar for the critic's loss, accounting for the relevant factors

    '''

    return -(torch.mean(crit_real_pred)-torch.mean(crit_fake_pred))+gp*c_lambda
cur_step = 0

generator_losses = []

critic_losses = []

for epoch in range(n_epochs):

    for real in tqdm(batches):

        cur_batch_size = len(real)

        real = real.to(device)

        mean_iteration_critic_loss = 0

        for _ in range(crit_repeats):

            ### Update critic ###

            crit_opt.zero_grad()

            fake_noise = get_noise(cur_batch_size, z_dim, device=device)

            fake = gen(fake_noise).detach()

            crit_fake_pred = crit(fake)

            crit_real_pred = crit(real)



            epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)

            gradient = get_gradient(crit, real, fake.detach(), epsilon)

            gp = gradient_penalty(gradient)

            crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)



            # Keep track of the average critic loss in this batch

            mean_iteration_critic_loss += crit_loss.item() / crit_repeats

            # Update gradients

            crit_loss.backward(retain_graph=True)

            # Update optimizer

            crit_opt.step()

        critic_losses += [mean_iteration_critic_loss]

        

        ### Update generator ###

        gen_opt.zero_grad()

        fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)

        fake_2 = gen(fake_noise_2)

        crit_fake_pred = crit(fake_2)

        

        gen_loss = get_gen_loss(crit_fake_pred)

        gen_loss.backward()



        # Update the weights

        gen_opt.step()



        # Keep track of the average generator loss

        generator_losses += [gen_loss.item()]

        

        ### Visualization code ###

        if cur_step % display_step == 0 and cur_step > 0:

            gen_mean = sum(generator_losses[-display_step:]) / display_step

            crit_mean = sum(critic_losses[-display_step:]) / display_step

            print(f"Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")

            show_tensor_images(fake)

            show_tensor_images(real)

            step_bins = 20

            num_examples = (len(generator_losses) // step_bins) * step_bins

            plt.plot(

                range(num_examples // step_bins), 

                torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),

                label="Generator Loss"

            )

            plt.plot(

                range(num_examples // step_bins), 

                torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),

                label="Critic Loss"

            )

            plt.legend()

            plt.show()



        cur_step += 1

gen.eval()

crit.eval()
torch.save(gen,'gen.pt')

torch.save(crit,'crit.pt')