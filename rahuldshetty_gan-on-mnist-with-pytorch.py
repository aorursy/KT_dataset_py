# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output
import os

import numpy as np

import errno

import torchvision.utils as vutils

from tensorboardX import SummaryWriter

from IPython import display

from matplotlib import pyplot as plt

import torch



'''

    TensorBoard Data will be stored in './runs' path

'''





class Logger:



    def __init__(self, model_name, data_name):

        self.model_name = model_name

        self.data_name = data_name



        self.comment = '{}_{}'.format(model_name, data_name)

        self.data_subdir = '{}/{}'.format(model_name, data_name)



        # TensorBoard

        self.writer = SummaryWriter(comment=self.comment)



    def log(self, d_error, g_error, epoch, n_batch, num_batches):



        # var_class = torch.autograd.variable.Variable

        if isinstance(d_error, torch.autograd.Variable):

            d_error = d_error.data.cpu().numpy()

        if isinstance(g_error, torch.autograd.Variable):

            g_error = g_error.data.cpu().numpy()



        step = Logger._step(epoch, n_batch, num_batches)

        self.writer.add_scalar(

            '{}/D_error'.format(self.comment), d_error, step)

        self.writer.add_scalar(

            '{}/G_error'.format(self.comment), g_error, step)



    def log_images(self, images, num_images, epoch, n_batch, num_batches, format='NCHW', normalize=True):

        '''

        input images are expected in format (NCHW)

        '''

        if type(images) == np.ndarray:

            images = torch.from_numpy(images)

        

        if format=='NHWC':

            images = images.transpose(1,3)

        



        step = Logger._step(epoch, n_batch, num_batches)

        img_name = '{}/images{}'.format(self.comment, '')



        # Make horizontal grid from image tensor

        horizontal_grid = vutils.make_grid(

            images, normalize=normalize, scale_each=True)

        # Make vertical grid from image tensor

        nrows = int(np.sqrt(num_images))

        grid = vutils.make_grid(

            images, nrow=nrows, normalize=True, scale_each=True)



        # Add horizontal images to tensorboard

        self.writer.add_image(img_name, horizontal_grid, step)



        # Save plots

        self.save_torch_images(horizontal_grid, grid, epoch, n_batch)



    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True):

        out_dir = './data/images/{}'.format(self.data_subdir)

        Logger._make_dir(out_dir)



        # Plot and save horizontal

        fig = plt.figure(figsize=(16, 16))

        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))

        plt.axis('off')

        if plot_horizontal:

            display.display(plt.gcf())

        self._save_images(fig, epoch, n_batch, 'hori')

        plt.close()



        # Save squared

        fig = plt.figure()

        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))

        plt.axis('off')

        self._save_images(fig, epoch, n_batch)

        plt.close()



    def _save_images(self, fig, epoch, n_batch, comment=''):

        out_dir = './data/images/{}'.format(self.data_subdir)

        Logger._make_dir(out_dir)

        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir,

                                                         comment, epoch, n_batch))



    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):

        

        # var_class = torch.autograd.variable.Variable

        if isinstance(d_error, torch.autograd.Variable):

            d_error = d_error.data.cpu().numpy()

        if isinstance(g_error, torch.autograd.Variable):

            g_error = g_error.data.cpu().numpy()

        if isinstance(d_pred_real, torch.autograd.Variable):

            d_pred_real = d_pred_real.data

        if isinstance(d_pred_fake, torch.autograd.Variable):

            d_pred_fake = d_pred_fake.data

        

        

        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(

            epoch,num_epochs, n_batch, num_batches)

             )

        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))

        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))



    def save_models(self, generator, discriminator, epoch):

        out_dir = './data/models/{}'.format(self.data_subdir)

        Logger._make_dir(out_dir)

        torch.save(generator.state_dict(),

                   '{}/G_epoch_{}'.format(out_dir, epoch))

        torch.save(discriminator.state_dict(),

                   '{}/D_epoch_{}'.format(out_dir, epoch))



    def close(self):

        self.writer.close()



    # Private Functionality



    @staticmethod

    def _step(epoch, n_batch, num_batches):

        return epoch * num_batches + n_batch



    @staticmethod

    def _make_dir(directory):

        try:

            os.makedirs(directory)

        except OSError as e:

            if e.errno != errno.EEXIST:

                raise
import torch

from torch import nn, optim

from torch.autograd.variable import Variable

from torchvision import transforms, datasets
def mnist_data():

    compose = transforms.Compose(

        [transforms.ToTensor(),

         transforms.Normalize((.5, ), (.5, ))

        ])

    out_dir = './dataset'

    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)# Load data

data = mnist_data()# Create loader with data, so that we can iterate over it

data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)

# Num batches

num_batches = len(data_loader)
print(data)
class Discriminator(torch.nn.Module):

    def __init__(self):

        super(Discriminator,self).__init__()

        self.hidden0 = nn.Sequential(

            nn.Linear(784,1024),

            nn.LeakyReLU(0.2),

            nn.Dropout(0.3)

        )

        self.hidden1 = nn.Sequential(

            nn.Linear(1024,512),

            nn.LeakyReLU(0.2),

            nn.Dropout(0.3)

        )

        self.hidden2 = nn.Sequential(

            nn.Linear(512,256),

            nn.LeakyReLU(0.2),

            nn.Dropout(0.3)

        )

        self.out = nn.Sequential(

            nn.Linear(256,1),

            nn.Sigmoid()

        )

        

    def forward(self,x):

        x = self.hidden0(x)

        x = self.hidden1(x)

        x = self.hidden2(x)

        x = self.out(x)

        return x

discriminator  = Discriminator()

print(discriminator)
def images_to_vectors(images):

    return images.view(-1,784)

def vectors_to_images(vectors):

    return vectors.view(-1,1,28,28)
class Generator(nn.Module):

    def __init__(self):

        super(Generator,self).__init__()

        self.hidden0 = nn.Sequential(

            nn.Linear(100,256),

            nn.LeakyReLU(0.2)

        )

        self.hidden1 = nn.Sequential(

            nn.Linear(256,512),

            nn.LeakyReLU(0.2)

        )

        self.hidden2 = nn.Sequential(

            nn.Linear(512,1024),

            nn.LeakyReLU(0.2)

        )

        self.out = nn.Sequential(

            nn.Linear(1024,784),

            nn.Tanh()

        )

        

    def forward(self,x):

        x = self.hidden0(x)

        x = self.hidden1(x)

        x = self.hidden2(x)

        x = self.out(x)

        return x



generator = Generator()

print(generator)
def noise(size):

    n = Variable(torch.randn(size,100))

    return n
d_optim = optim.Adam(discriminator.parameters(),lr = 0.0002 )

g_optim = optim.Adam(generator.parameters(),lr = 0.0002 )
loss = nn.BCELoss()
def ones_target(size):

    return Variable(torch.ones(size,1))



def zeros_target(size):

    return Variable(torch.zeros(size,1))
def train_discriminator(optimizer,real_data,fake_data):

    N = real_data.size(0)

    optimizer.zero_grad()

    

    prediction_real = discriminator(real_data)

    error_real = loss(prediction_real,ones_target(N))

    error_real.backward()

    

    prediction_fake = discriminator(fake_data)

    error_fake = loss(prediction_fake,zeros_target(N))

    error_fake.backward()

    

    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake
def train_generator(optimizer,fake_data):

    N = fake_data.size(0)

    optimizer.zero_grad()

    prediction = discriminator(fake_data)

    error = loss(prediction,ones_target(N))

    error.backward()

    optimizer.step()

    return error
num_test_samples = 16

test_noise = noise(num_test_samples)
num_epochs = 100



logger = Logger(model_name='VGAN', data_name='MNIST')



for epoch in range(num_epochs):

    for n_batch, (real_batch,_) in enumerate(data_loader):

        N = real_batch.size(0)

        

        # train the discriminator

        real_data = Variable(images_to_vectors(real_batch))

        

        fake_data = generator(noise(N)).detach()

        

        d_error,d_pred_real,d_pred_fake = train_discriminator(d_optim,real_data,fake_data)

        

        # traing the generator

        fake_data = generator(noise(N))

        g_error = train_generator(g_optim,fake_data)

        

        logger.log(d_error, g_error, epoch, n_batch, num_batches)

        

        if (n_batch) % 300 == 0 and n_batch != 0 and epoch % 20 == 0:

            test_images = vectors_to_images(generator(test_noise))

            test_images = test_images.data

            

            logger.log_images(

                test_images, num_test_samples, epoch,n_batch,num_batches

            )

            logger.display_status(

                epoch,num_epochs,n_batch,num_batches,d_error,g_error,d_pred_real,d_pred_fake

            )
torch.save(generator.state_dict(), 'genrator.pt')

torch.save(discriminator.state_dict(), 'discriminator.pt')