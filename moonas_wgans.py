#based on https://github.com/shayneobrien/generative-models



import math

import torch, torchvision

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.autograd import Variable



import os

import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import multivariate_normal



from itertools import product

from tqdm import tqdm



#import wgan_base as wb
import math

import torch, torchvision

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.autograd import Variable



import os

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import numpy as np

#import seaborn as sns; sns.set(color_codes=True)

from scipy.stats import kde



from itertools import product

from tqdm import tqdm



import torchvision.datasets as datasets

import torchvision.transforms as transforms



plt.style.use('ggplot')

plt.rcParams["figure.figsize"] = (8,6)



def sep(data):

    if(torch.is_tensor(data)):

        d = data.cpu().detach().numpy()

    else:

        d = data

    x = [d[i][0] for i in range(len(d))]

    y = [d[i][1] for i in range(len(d))]

    return x,y



def to_var(x):

    """ Make a tensor cuda-erized and requires gradient """

    return to_cuda(x).requires_grad_()



def to_cuda(x):

    """ Cuda-erize a tensor """

    if torch.cuda.is_available():

        x = x.cuda()

    return x



def get_dataloader(data, BATCH_SIZE=64, tt_split = 0.8):

    """ Load data for binared MNIST """

    

    

    # split randomized data into train:split*0.9,    val: split*0.1,    test: 1-split

    train_size = len(data)*tt_split

    train_dataset = torch.tensor(data[:int(np.ceil(train_size*0.9))]).float()

    val_dataset = torch.tensor(data[int(np.ceil(train_size*0.9)):int(np.ceil(train_size))]).float()

    test_dataset = torch.tensor(data[int(np.ceil(train_size)):]).float()

    

    train_dataset = to_cuda(train_dataset)

    val_dataset = to_cuda(val_dataset)

    test_dataset = to_cuda(test_dataset)



    # Create data loaders

    train = torch.utils.data.TensorDataset(train_dataset, torch.zeros(train_dataset.shape[0]))

    val = torch.utils.data.TensorDataset(val_dataset, torch.zeros(val_dataset.shape[0]))

    test = torch.utils.data.TensorDataset(test_dataset, torch.zeros(test_dataset.shape[0]))



    train_iter = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

    val_iter = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)

    test_iter = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)



    return train_iter, val_iter, test_iter



# Style plots (decorator)

def plot_styling(func):

    def wrapper(*args, **kwargs):

        style = {'axes.titlesize': 24,

                 'axes.labelsize': 20,

                 'lines.linewidth': 3,

                 'lines.markersize': 10,

                 'xtick.labelsize': 16,

                 'ytick.labelsize': 16,

                 'panel.background': element_rect(fill="white"),

                 'panel.grid.major': element_line(colour="grey50"),

                 'panel.grid.minor': element_line(colour="grey50")

                }

        with plt.style.context((style)):

            ax = func(*args, **kwargs)      

    return wrapper



class Generator(nn.Module):

    """ Generator. Input is noise, output is a generated image.

    """

    def __init__(self, image_size, hidden_dim, n_layers, z_dim):

        super(Generator, self).__init__()



        self.inputlayer = nn.Linear(z_dim, hidden_dim)

        self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers)])

        self.generate = nn.Linear(hidden_dim, image_size)



    def forward(self, x):

        x = F.relu(self.inputlayer(x))

        for l in self.linears:

            x = F.relu(l(x))

        x = self.generate(x)           # TODO : torch.sigmoid?

        return x

        



class Discriminator(nn.Module):

    """ Critic (not trained to classify). Input is an image (real or generated),

    output is the approximate Wasserstein Distance between z~P(G(z)) and real.

    """

    def __init__(self, image_size, hidden_dim, n_layers, output_dim):

        super(Discriminator, self).__init__()

        """   batchnorm

        self.linears = nn.ModuleList([nn.Linear(image_size, hidden_dim), nn.BatchNorm1d(hidden_dim)])

        for i in range(n_layers):

            self.linears.extend([nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)])

        """

        self.inputlayer = nn.Linear(image_size, hidden_dim)

        self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers)])

        self.discriminate = nn.Linear(hidden_dim, output_dim)



    def forward(self, x):

        """  batchnorm

        for i, l in enumerate(self.linears):

            if i % 2 == 0:

                x = self.linears[i+1](F.relu(l(x)))

        x = self.discriminate(x)

        return x

        

        """

        x = F.relu(self.inputlayer(x))

        for l in self.linears:

            x = F.relu(l(x))

        x = self.discriminate(x)   # TODO : torch.sigmoid?

        return x

    

class vanilla_Discriminator(nn.Module):

    """ Critic (not trained to classify). Input is an image (real or generated),

    output is the approximate Wasserstein Distance between z~P(G(z)) and real.

    """

    def __init__(self, image_size, hidden_dim, n_layers, output_dim):

        super(vanilla_Discriminator, self).__init__()

        """   batchnorm

        self.linears = nn.ModuleList([nn.Linear(image_size, hidden_dim), nn.BatchNorm1d(hidden_dim)])

        for i in range(n_layers):

            self.linears.extend([nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim)])

        """

        self.inputlayer = nn.Linear(image_size, hidden_dim)

        self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers)])

        self.discriminate = nn.Linear(hidden_dim, output_dim)



    def forward(self, x):

        """  batchnorm

        for i, l in enumerate(self.linears):

            if i % 2 == 0:

                x = self.linears[i+1](F.relu(l(x)))

        x = self.discriminate(x)

        return x

        

        """

        x = F.relu(self.inputlayer(x))

        for l in self.linears:

            x = F.relu(l(x))

        x = torch.sigmoid(self.discriminate(x))  # TODO : torch.sigmoid?

        return x

    



class WGAN(nn.Module):

    """ Super class to contain both Discriminator (D) and Generator (G)

    """

    def __init__(self, image_size, hidden_dim, n_layers, z_dim, output_dim=1):

        super().__init__()



        self.__dict__.update(locals())



        self.G = Generator(image_size, hidden_dim, n_layers, z_dim)

        self.D = Discriminator(image_size, hidden_dim, n_layers, output_dim)



        self.shape = int(image_size ** 0.5)

        

class GAN(nn.Module):

    """ Super class to contain both Discriminator (D) and Generator (G)

    """

    def __init__(self, image_size, hidden_dim, n_layers, z_dim, output_dim=1):

        super().__init__()



        self.__dict__.update(locals())



        self.G = Generator(image_size, hidden_dim, n_layers, z_dim)

        self.D = vanilla_Discriminator(image_size, hidden_dim, n_layers, output_dim)



        self.shape = int(image_size ** 0.5)





class GANTrainer:

    """ Object to hold data iterators, train a GAN variant

    """

    def __init__(self, model, train_iter, val_iter, test_iter, viz=False, gantype = 'wgangp'):

        self.model = to_cuda(model)

        self.name = model.__class__.__name__



        self.train_iter = train_iter

        self.val_iter = val_iter

        self.test_iter = test_iter



        self.Glosses = []

        self.Dlosses = []



        self.viz = viz

        self.num_epochs = 0

        self.gantype = gantype

        self.G_iter = 0



    def train(self, num_epochs, G_lr=5e-5, D_lr=5e-5, G_wd = 0, D_steps_standard=5, clip=0.01, G_init=5, G_per_D = 1):

        """ Train a Wasserstein GAN

            Logs progress using G loss, D loss, G(x), D(G(x)), visualizations

            of Generator output.

        Inputs:

            num_epochs: int, number of epochs to train for

            G_lr: float, learning rate for generator's RMProp optimizer

            D_lr: float, learning rate for discriminator's RMSProp optimizer

            D_steps: int, ratio for how often to train D compared to G

            clip: float, bound for parameters [-c, c] to enforce K-Lipschitz

        """

        # Initialize optimizers

        if(self.gantype in ['wgan', 'wgangp', 'wganlp', 'ls']):

            bet = (0.5,0.9)

        else:

            bet = (0.5, 0.9)

        

        G_optimizer = optim.Adam(params=[p for p in self.model.G.parameters()

                                        if p.requires_grad], lr=G_lr, weight_decay = G_wd,betas=bet)

        D_optimizer = optim.Adam(params=[p for p in self.model.D.parameters()

                                        if p.requires_grad], lr=D_lr,betas=bet)



        num_batches = len(self.train_iter)

        D_steps = D_steps_standard

        self.model.train()

        if(self.gantype in ['gan','nsgan']):

                self.pretrain(G_init, G_optimizer)



        # Begin training

        for epoch in tqdm(range(1, num_epochs+1)):

            #Train discriminator to almost convergence to approx W

            if(self.gantype in ['wgangp', 'wgan','wganlp']):

                if( (self.G_iter <= 25) or (self.G_iter % 100 == 0) ):

                    D_steps = 100

                else:               

                    D_steps = D_steps_standard

            else:

                D_steps = D_steps_standard

                    



            self.model.train()

            G_losses, D_losses = [], []

            ep_iter = 0



            while(ep_iter < num_batches):



                if(self.G_iter % G_per_D == 0):

                    D_step_loss = []



                    for _ in range(D_steps):



                        # Reshape images

                        images = self.process_batch(self.train_iter)



                        # TRAINING D: Zero out gradients for D

                        D_optimizer.zero_grad()



                        # Train the discriminator to approximate the Wasserstein

                        # distance between real, generated distributions

                        if(self.gantype == 'wgangp'):

                            D_loss = self.train_D_GP(images)

                        elif(self.gantype == 'wganlp'):

                            D_loss = self.train_D_LP(images, LAMBDA = 1)

                        elif(self.gantype == 'wgan'):

                            D_loss = self.train_D_W(images)

                        elif(self.gantype in ['nsgan','gan']):

                                D_loss = self.train_D_vanilla(images)

                        elif(self.gantype == 'ls'):

                            D_loss = self.train_D_ls(images)

                        else:

                            print('Unknown gantype')

                            break





                        # Update parameters

                        D_loss.backward()

                        D_optimizer.step()



                        # Log results, backpropagate the discriminator network

                        D_step_loss.append(D_loss.item())

                        ep_iter += 1



                        if(self.gantype == 'wgan'):

                            # Clamp weights (crudely enforces K-Lipschitz)

                            self.clip_D_weights(clip)



                    # We report D_loss in this way so that G_loss and D_loss have

                    # the same number of entries.

                    D_losses.append(np.mean(D_step_loss))

                else:

                    images = self.process_batch(self.train_iter)



                '''

                # Visualize generator progress

                if self.viz:

                    if(self.G_iter < 200 and self.G_iter % 10 == 0):

                        self.viz_data(save = True)

                    elif(self.G_iter >200 and self.G_iter % 200 == 0):

                        self.viz_data(save=True)

                '''

                # TRAINING G: Zero out gradients for G

                G_optimizer.zero_grad()



                if(self.gantype in ['wgan', 'wgangp', 'wganlp']):

                    # Train the generator to (roughly) minimize the approximated

                    # Wasserstein distance

                    G_loss = self.train_G_W(images)

                elif(self.gantype == 'gan'):

                    G_loss = self.train_G_vanilla(images)

                elif(self.gantype == 'nsgan'):

                    G_loss = self.train_G_ns(images)

                elif(self.gantype == 'ls'):

                    G_loss = self.train_G_ls(images)

                else:

                    print('Unknown gantype')

                    break



                # Log results, update parameters

                G_losses.append(G_loss.item())

                G_loss.backward()

                G_optimizer.step()

                self.G_iter += 1

                

            



            # Save progress

            if self.viz:

                    if(self.num_epochs <= 30 and self.num_epochs % 5 == 0):

                        self.viz_data(save = True)

                    elif(self.num_epochs > 30 and self.num_epochs < 101 and self.num_epochs % 10 == 0):

                        self.viz_data(save = True)

                    elif(self.num_epochs > 100 and self.num_epochs % 20 == 0):

                        self.viz_data(save = True)

            self.Glosses.extend(G_losses)

            self.Dlosses.extend(D_losses)

            self.num_epochs += 1



            """

            # Progress logging

            print ("Epoch[%d/%d], G Loss: %.4f, D Loss: %.4f"

                   %(epoch, num_epochs, np.mean(G_losses), np.mean(D_losses)))

            self.num_epochs += 1

            """

            

            

    

    def train_D_GP(self, images, LAMBDA=0.1):

        """ Run 1 step of training for discriminator

        Input:

            images: batch of images (reshaped to [batch_size, -1])

        Output:

            D_loss: Wasserstein loss for discriminator,

            -E[D(x)] + E[D(G(z))] + λE[(||∇ D(εx + (1 − εG(z)))|| - 1)^2]

        """

        # ORIGINAL CRITIC STEPS:

        # Sample noise, an output from the generator

        noise = self.compute_noise(images.shape[0], self.model.z_dim)

        G_output = self.model.G(noise)



        # Use the discriminator to sample real, generated images

        DX_score = self.model.D(images) # D(z)

        DG_score = self.model.D(G_output) # D(G(z))



        # GRADIENT PENALTY:

        # Uniformly sample along one straight line per each batch entry.

        epsilon = to_var(torch.rand(images.shape[0], 1).expand(images.size()))



        # Generate images from the noise, ensure unit gradient norm 1

        # See Section 4 and Algorithm 1 of original paper for full explanation.

        G_interpolation = epsilon*images + (1-epsilon)*G_output

        D_interpolation = self.model.D(G_interpolation)



        # Compute the gradients of D with respect to the noise generated input

        weight = to_cuda(torch.ones(D_interpolation.size()))



        gradients = torch.autograd.grad(outputs=D_interpolation,

                                        inputs=G_interpolation,

                                        grad_outputs=weight,

                                        only_inputs=True,

                                        create_graph=True,

                                        retain_graph=True)[0]



        # Full gradient penalty

        grad_penalty = LAMBDA * torch.mean((gradients.norm(2, dim=1) - 1)**2)



        # Compute WGAN-GP loss for D

        D_loss = torch.mean(DG_score) - torch.mean(DX_score) + grad_penalty



        return D_loss

    

    def train_D_LP(self, images, LAMBDA=0.1):

        """ Run 1 step of training for discriminator

        Input:

            images: batch of images (reshaped to [batch_size, -1])

        Output:

            D_loss: Wasserstein loss for discriminator,

            -E[D(x)] + E[D(G(z))] + λE[max(0,∇ D(εx + (1 − εG(z))) - 1)^2]

        """

        # ORIGINAL CRITIC STEPS:

        # Sample noise, an output from the generator

        noise = self.compute_noise(images.shape[0], self.model.z_dim)

        G_output = self.model.G(noise)



        # Use the discriminator to sample real, generated images

        DX_score = self.model.D(images) # D(z)

        DG_score = self.model.D(G_output) # D(G(z))



        # GRADIENT PENALTY:

        # Uniformly sample along one straight line per each batch entry.

        epsilon = to_var(torch.rand(images.shape[0], 1).expand(images.size()))



        # Generate images from the noise, ensure unit gradient norm 1

        # See Section 4 and Algorithm 1 of original paper for full explanation.

        G_interpolation = epsilon*images + (1-epsilon)*G_output

        D_interpolation = self.model.D(G_interpolation)



        # Compute the gradients of D with respect to the noise generated input

        weight = to_cuda(torch.ones(D_interpolation.size()))



        gradients = torch.autograd.grad(outputs=D_interpolation,

                                        inputs=G_interpolation,

                                        grad_outputs=weight,

                                        only_inputs=True,

                                        create_graph=True,

                                        retain_graph=True)[0]



        # Full gradient penalty

        zer = torch.zeros(gradients.norm(2,dim=1).shape[0])

        grad_penalty = LAMBDA * torch.mean((torch.max(zer, gradients.norm(2, dim=1) - 1))**2)



        # Compute WGAN-GP loss for D

        D_loss = torch.mean(DG_score) - torch.mean(DX_score) + grad_penalty



        return D_loss

    



    def train_D_W(self, images):

        """ Run 1 step of training for discriminator

        Input:

            images: batch of images (reshaped to [batch_size, -1])

        Output:

            D_loss: wasserstein loss for discriminator,

            -E[D(x)] + E[D(G(z))]

        """

        # Sample from the generator

        noise = self.compute_noise(images.shape[0], self.model.z_dim)

        G_output = self.model.G(noise)



        # Score real, generated images

        DX_score = self.model.D(images) # D(x), "real"

        DG_score = self.model.D(G_output) # D(G(x')), "fake"



        # Compute WGAN loss for D

        D_loss = -1 * (torch.mean(DX_score)) + torch.mean(DG_score)



        return D_loss

    

    def train_D_vanilla(self, images):

        """ Run 1 step of training for discriminator

        Input:

            images: batch of images (reshaped to [batch_size, -1])

        Output:

            D_loss: non-saturing loss for discriminator,

            -E[log(D(x))] - E[log(1 - D(G(z)))]

        """



        # Sample noise z, generate output G(z)

        noise = self.compute_noise(images.shape[0], self.model.z_dim)

        G_output = self.model.G(noise)



        # Classify the generated and real batch images

        DX_score = self.model.D(images) # D(x)

        DG_score = self.model.D(G_output) # D(G(z))



        # Compute vanilla (original paper) D loss

        D_loss = -torch.mean(torch.log(DX_score + 1e-8)) + torch.mean(torch.log(1 - DG_score + 1e-8))

        return D_loss

    

    def train_D_ls(self, images):

        # Sample noise z, generate output G(z)

        noise = self.compute_noise(images.shape[0], self.model.z_dim)

        G_output = self.model.G(noise)



        # Classify the generated and real batch images

        DX_score = self.model.D(images) # D(x)

        DG_score = self.model.D(G_output) # D(G(z))

        

        D_loss = 0.5 * (torch.mean(torch.square(DX_score-1)) + torch.mean(torch.square(DG_score)))

        return D_loss



    def train_G_W(self, images):

        """ Run 1 step of training for generator

        Input:

            images: batch of images (reshaped to [batch_size, -1])

        Output:

            G_loss: wasserstein loss for generator,

            -E[D(G(z))]

        """

        # Get noise, classify it using G, then classify the output of G using D.

        noise = self.compute_noise(images.shape[0], self.model.z_dim) # z

        G_output = self.model.G(noise) # G(z)

        DG_score = self.model.D(G_output) # D(G(z))



        # Compute WGAN loss for G

        G_loss = -1 * (torch.mean(DG_score))



        return G_loss

    



    def train_G_ns(self, images):

        """ Run 1 step of training for generator

        Input:

            images: batch of images reshaped to [batch_size, -1]

        Output:

            G_loss: non-saturating loss for how well G(z) fools D,

            -E[log(D(G(z)))]

        """



        # Get noise (denoted z), classify it using G, then classify the output

        # of G using D.

        noise = self.compute_noise(images.shape[0], self.model.z_dim) # (z)

        G_output = self.model.G(noise) # G(z)

        DG_score = self.model.D(G_output) # D(G(z))



        # Compute the non-saturating loss for how D did versus the generations

        # of G using sigmoid cross entropy

        G_loss = -torch.mean(torch.log(DG_score + 1e-8))



        return G_loss

    

    def train_G_vanilla(self, images):

        """ Run 1 step of training for generator

        Input:

            images: batch of images reshaped to [batch_size, -1]

        Output:

            G_loss: minimax loss for how well G(z) fools D,

            E[log(1-D(G(z)))]

        """

        # Get noise (denoted z), classify it using G, then classify the output of G using D.

        noise = self.compute_noise(images.shape[0], self.model.z_dim) # z

        G_output = self.model.G(noise) # G(z)

        DG_score = self.model.D(G_output) # D(G(z))



        # Compute the minimax loss for how D did versus the generations of G using sigmoid cross entropy

        G_loss = torch.mean(torch.log((1-DG_score) + 1e-8))



        return G_loss

    

    def train_G_ls(self, images):

        # Sample noise z, generate output G(z)

        noise = self.compute_noise(images.shape[0], self.model.z_dim)

        G_output = self.model.G(noise)



        # Classify the generated and real batch images

        DG_score = self.model.D(G_output) # D(G(z))

        

        G_loss = 0.5 * torch.mean(torch.square(DG_score-1))

        return G_loss

    

    def pretrain(self, G_init, G_optimizer):

        # Let G train for a few steps before beginning to jointly train G

        # and D because MM GANs have trouble learning very early on in training

        if G_init > 0:

            for _ in range(G_init):

                # Process a batch of images

                images = self.process_batch(self.train_iter)



                # Zero out gradients for G

                G_optimizer.zero_grad()



                # Pre-train G

                G_loss = self.train_G_vanilla(images)



                # Backpropagate the generator network

                G_loss.backward()

                G_optimizer.step()



            print('G pre-trained for {0} training steps.'.format(G_init))

        else:

            print('G not pre-trained -- GAN unlikely to converge.')



    def compute_noise(self, batch_size, z_dim):

        """ Compute random noise for input into the Generator G """

        return to_cuda(torch.randn(batch_size, z_dim))



    def process_batch(self, iterator):

        """ Generate a process batch to be input into the Discriminator D """

        images, _ = next(iter(iterator))

        images = to_cuda(images.view(images.shape[0], -1))

        return images



    def clip_D_weights(self, clip):

        for parameter in self.model.D.parameters():

            parameter.data.clamp_(-clip, clip)



    def generate_images(self, epoch=-1, num_outputs=50):

        """ Visualize progress of generator learning """

        # Turn off any regularization

        self.model.eval()



        # Sample noise vector

        noise = self.compute_noise(num_outputs, self.model.z_dim)



        # Transform noise to image

        images = self.model.G(noise)

        

        return images



        """

        # Reshape to square image size

        images = images.view(images.shape[0],

                             self.model.shape,

                             self.model.shape,

                             -1).squeeze()



        # Plot

        plt.close()

        grid_size, k = int(num_outputs**0.5), 0

        fig, ax = plt.subplots(grid_size, grid_size, figsize=(5, 5))

        for i, j in product(range(grid_size), range(grid_size)):

            ax[i,j].get_xaxis().set_visible(False)

            ax[i,j].get_yaxis().set_visible(False)

            ax[i,j].imshow(images[k].data.numpy(), cmap='gray')

            k += 1





        # Save images if desired

        if save:

            outname = '../viz/' + self.name + '/'

            if not os.path.exists(outname):

                os.makedirs(outname)

            torchvision.utils.save_image(images.unsqueeze(1).data,

                                         outname + 'reconst_%d.png'

                                         %(epoch), nrow=grid_size)

        """



    def viz_loss(self, save = False):

        """ Visualize loss for the generator, discriminator """

        # Set style, figure size

        plt.style.use('ggplot')

        plt.rcParams["figure.figsize"] = (8,6)



        # Plot Discriminator loss in red

        plt.plot(np.linspace(1, self.G_iter, len(self.Dlosses)),

                 self.Dlosses,

                 'b')

        print(self.num_epochs,self.G_iter, len(self.Dlosses))



        """

        # Plot Generator loss in green

        plt.plot(np.linspace(1, self.num_epochs, len(self.Dlosses)),

                 self.Glosses,

                 'g')



        # Add legend, title

        plt.legend(['Discriminator', 'Generator'])

        """

        #plt.title(self.name)

        if save:

            plt.savefig('loss_%s_%d.png'%(self.gantype, self.G_iter), dpi = 150)

        plt.show()

        

        

    def viz_data(self, save = False, density=True):

        if(density == True):

            lower = (-1.3, -1.3)

            upper = (1.3, 1.3)

            nbins=300

            x,y = sep(self.generate_images(num_outputs = 1000))

            k = kde.gaussian_kde([x,y])

            xi, yi = np.mgrid[lower[0]:upper[1]:nbins*1j, lower[1]:upper[1]:nbins*1j]

            zi = k(np.vstack([xi.flatten(), yi.flatten()]))



            fig, ax = plt.subplots()



            # a (1-alpha)-confidence circle for 2d gaussian with cov = a * eyes has radius sqrt(-2 a ln(alpha)), here a = 3/400

            ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Blues)

            for i in range(len(means)):

                circle = plt.Circle((means[i][0]/10, means[i][1]/10),           # (x,y)

                np.sqrt(-3 * np.log(0.05)/200), color='black',linewidth = 2, fill=False)

                ax.add_artist(circle)

            #plt.colorbar()

            

        else:

            # 2-dim points rowwise concatenated in array

            # Set style, figure size

            plt.style.use('ggplot')

            plt.rcParams["figure.figsize"] = (8,6)



            # Plot generated points in red

            gen = self.generate_images()

            xg = [gen[i][0] for i in range(len(gen))]

            yg = [gen[i][1] for i in range(len(gen))]

            plt.plot(xg,

                     yg,

                     'ro', markersize = 3)





            # Plot real data points in green

            real = next(iter(train_iter))[0]

            xr = [real[i][0] for i in range(len(real))]

            yr = [real[i][1] for i in range(len(real))]

            plt.plot(xr,

                     yr,

                     'go', markersize = 3)



            # Add legend, title

            plt.legend(['generated', 'real'])

        

        #plt.title(self.name)

        if save:

            plt.savefig('data_%s_%d_%d.png'%(self.gantype,self.num_epochs, self.G_iter), dpi = 150)

        plt.show()

        



    def save_model(self):

        """ Save model state dictionary """

        torch.save(self.model.state_dict(), 'model_%s_%d'%(self.gantype, self.G_iter))



    def load_model(self, loadpath):

        """ Load state dictionary into model """

        state = torch.load(loadpath)

        self.model.load_state_dict(state)
images = torch.tensor([[2,1],[3,4],[5,7]])



torch.rand(images.shape[0], 1).expand(images.size())
#create circle of Normals data

n_groups = 8

n_pergroup= 1000

resc = ''



angles = np.arange(0, 2*math.pi, step = 2*math.pi / n_groups)

means = [[10*np.cos(i), 10*np.sin(i)] for i in angles]





data = []

for i in range(n_groups):

    data.extend(multivariate_normal.rvs(means[i], cov = 0.75 * np.eye(2), size = n_pergroup))

    

data = np.array(data)

if(resc == 'comp'):

    # rescale to [0,1]^2

    scales = np.amax(data, axis = 0)-np.amin(data,axis = 0)

    data = data - np.amin(data,axis = 0)

    data[:,0] = data[:,0] / scales[0]

    data[:,1] = data[:,1] / scales[1]

else:

    data /= 10



#shuffle rows

np.random.shuffle(data)

plt.style.use('ggplot')

plt.rcParams["figure.figsize"] = (8,6)

x,y = sep(data)

fig1 = plt.gcf()

plt.plot(x,y, 'bo', markersize = 3)

fig1.savefig('orig_data', dpi = 150)
# see clip below, original architecture reaches magnitude of values as Lip1 functions

(1/(2 * np.sqrt(2)*512 ** 4)) ** (1/5) # close to 0.01...
torch.manual_seed(34)

# separate data into data loaders

train_iter, val_iter, test_iter = get_dataloader(data)



gantypes = ['gan', 'nsgan', 'ls', 'wgan', 'wgangp', 'wganlp']

gantypes = [ 'gan', 'nsgan']

clip =0.01

Dlosses=[]

Glosses=[]





for gantype in gantypes:

    if(gantype in ['gan', 'nsgan']):

        # Init model

        model = GAN(image_size=2, hidden_dim=256, n_layers = 3, z_dim=10)

    else:

        model = WGAN(image_size=2, hidden_dim=256, n_layers = 3, z_dim=10)



        

    if(gantype == 'wgan'):

        # choose good clip for architecture: d,p,L+1 (range should be +- sqrt(d)/2 and is +-c**(L+1) d p**L )

        L = model.n_layers+1

        d = model.image_size

        p = model.hidden_dim

        clip = 2 * (1/(2 * np.sqrt(d)* p ** L))**(1/(L+1))



    # Init trainer

    trainer = GANTrainer(model=model, train_iter=train_iter, val_iter=val_iter, test_iter=test_iter, viz=True, gantype = gantype)



    G_lr = 1e-4

    D_lr = 1e-4

    G_wd = 0

    if(gantype in ['gan','nsgan']):

        perD = 2

        G_lr = 1e-5

        D_lr = 1e-5



    if(gantype == 'wgan'):

        G_lr = 5e-5

        D_lr = 5e-5

        G_wd = 0.01

        #new: GAN, NSGAN same LR

    

    if(gantype in ['wgan', 'wgangp','wganlp']):

        D_steps_standard = 5

    else:

        D_steps_standard = 1





    # Train

    trainer.train(num_epochs=100, G_lr=G_lr, D_lr=D_lr,G_wd = G_wd, D_steps_standard=D_steps_standard, clip=clip, G_init = 20, G_per_D = perD)

    trainer.viz_loss(save = True)

    trainer.viz_data()

    trainer.train(num_epochs=200, G_lr=1e-4, D_lr=1e-4,G_wd = 0, D_steps_standard=D_steps_standard, clip=clip)

    trainer.viz_loss(save = True),

    trainer.viz_data(save = True)

    trainer.save_model()

    Dlosses.append(trainer.Dlosses)

    Glosses.append(trainer.Glosses)

    

torch.save(Dlosses, 'Dlosses')

torch.save(Glosses, 'Glosses')
# looks too linear (does relu work??? yes), prefers small scale



# discriminator doesn't learn, loss too close to 0,   -loss != W(P_r,P_g)



# too little training? no, bad clips -> batch norm



# normalize losses for suitable gradients (change for architecture and clip)



# for wc use batchnorm -> then implicitly not one to one function, but batch to batch and Lipschitz constant depends on batch



# gp recommend layer norm



# wc: 5e-5, batchnorm, standard adam?     gp: 1e-4, paper: beta = (0,0.9), implem: (0.5,0.9), wc and lsgan: RMSProp 5e-5, 1e-4 respectively (GP paper)



# if batchnorm or layernorm, use 0.001 wd in discriminator (GP paper)