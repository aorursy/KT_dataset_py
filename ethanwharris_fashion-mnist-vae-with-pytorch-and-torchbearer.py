!pip install torchbearer
import os



import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



import torchvision

from torchvision import transforms

from torchvision.utils import make_grid

from torchvision.datasets import FashionMNIST



import torchbearer

import torchbearer.callbacks as callbacks

from torchbearer import Trial, state_key



MU = state_key('mu')

LOGVAR = state_key('logvar')
class VAE(nn.Module):

    def __init__(self, latent_size):

        super(VAE, self).__init__()

        self.latent_size = latent_size



        self.encoder = nn.Sequential(

            nn.Conv2d(1, 32, 4, 1, 2),   # B,  32, 28, 28

            nn.ReLU(True),

            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14

            nn.ReLU(True),

            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7

        )

        

        self.mu = nn.Linear(64 * 7 * 7, latent_size)

        self.logvar = nn.Linear(64 * 7 * 7, latent_size)

        

        self.upsample = nn.Linear(latent_size, 64 * 7 * 7)

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  64,  14,  14

            nn.ReLU(True),

            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1), # B,  32, 28, 28

            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, 4, 1, 2)   # B, 1, 28, 28

        )



    def reparameterize(self, mu, logvar):

        if self.training:

            std = torch.exp(0.5*logvar)

            eps = torch.randn_like(std)

            return eps.mul(std).add_(mu)

        else:

            return mu



    def forward(self, x, state):

        image = x

        x = self.encoder(x).relu().view(x.size(0), -1)

        

        mu = self.mu(x)

        logvar = self.logvar(x)

        sample = self.reparameterize(mu, logvar)

        

        result = self.decoder(self.upsample(sample).relu().view(-1, 64, 7, 7))

        

        if state is not None:

            state[torchbearer.Y_TRUE] = image

            state[MU] = mu

            state[LOGVAR] = logvar

        

        return result
# Hack to make this work with the torchvision dataset



import codecs



def get_int(b):

    return int(codecs.encode(b, 'hex'), 16)



def read_label_file(path):

    with open(path, 'rb') as f:

        data = f.read()

        assert get_int(data[:4]) == 2049

        length = get_int(data[4:8])

        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)

        return torch.from_numpy(parsed).view(length).long()





def read_image_file(path):

    with open(path, 'rb') as f:

        data = f.read()

        assert get_int(data[:4]) == 2051

        length = get_int(data[4:8])

        num_rows = get_int(data[8:12])

        num_cols = get_int(data[12:16])

        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)

        return torch.from_numpy(parsed).view(length, num_rows, num_cols)



root = '../input'

out = '../processed'

if not os.path.exists(out):

    os.mkdir(out)



training_set = (

    read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),

    read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))

)

test_set = (

    read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),

    read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))

)

with open(os.path.join(out, 'training.pt'), 'wb') as f:

    torch.save(training_set, f)

with open(os.path.join(out, 'test.pt'), 'wb') as f:

    torch.save(test_set, f)
transform = transforms.Compose([transforms.ToTensor()])  # No augmentation

trainset = FashionMNIST(root='../', train=True, transform=transform)

testset = FashionMNIST(root='../', train=False, transform=transform)

traingen = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

testgen = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)
def beta_kl(mu_key, logvar_key, beta=5):

    @callbacks.add_to_loss

    def callback(state):

        mu = state[mu_key]

        logvar = state[logvar_key]

        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * beta

    return callback
def plot_progress(key=torchbearer.Y_PRED, num_images=100, nrow=10):

    @callbacks.on_step_validation

    @callbacks.once_per_epoch

    def callback(state):

        images = state[key]

        image = make_grid(images[:num_images], nrow=nrow, normalize=True)[0, :, :]

        plt.imshow(image.detach().cpu().numpy(), cmap="gray")

        plt.show()

    

    return callback
model = VAE(latent_size=10)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)

trial = Trial(model, optimizer, nn.MSELoss(reduction='sum'), metrics=['acc', 'loss'], callbacks=[

    beta_kl(MU, LOGVAR),

    callbacks.ConsolePrinter(),

    plot_progress()

], verbose=1).with_generators(train_generator=traingen, test_generator=testgen)

trial.to('cuda')

trial.run(20)

trial.evaluate(verbose=0, data_key=torchbearer.TEST_DATA)
from random import shuffle

SAMPLES = state_key('samples')

dims = [i for i in range(model.latent_size)]



def draw_space(self, x, state=None):

    image = x

    x = self.encoder(x).relu().view(x.size(0), -1)

    sample = self.mu(x)



    samples = sample.unsqueeze(1).repeat(1, 10, 1)

    space = torch.linspace(-3, 3, steps=10, device=x.device)

    

    for dim in dims:

        samples[dim, :, dim] = space



    sample = samples.view(-1, samples.size(2))

    

    result = self.decoder(self.upsample(sample).relu().view(-1, 64, 7, 7))

        

    if state is not None:

        state[SAMPLES] = result

    return None



import types

model.forward = types.MethodType(draw_space, model)  # HACK!
trial = Trial(model, callbacks=[

    plot_progress(SAMPLES, num_images=100, nrow=10)

], verbose=1).with_generators(test_generator=testgen)

trial.to('cuda')

_ = trial.for_test_steps(1).evaluate(data_key=torchbearer.TEST_DATA)