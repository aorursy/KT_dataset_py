import numpy as np

import pandas as pd

import sklearn as skl

import matplotlib.pyplot as plt



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch import optim

from torch.utils.data import Dataset
class MNIST(Dataset):

    def __init__(self, filename, batch_size, train=True, shuffle=None):

        data = pd.read_csv(filename)

        

        self.batch_size = batch_size



        if shuffle is None:

            shuffle = train

        

        if shuffle:

            data = skl.utils.shuffle(data)

            data.reset_index(inplace=True, drop=True)



        if train:

            self.images = data.iloc[:, 1:] / 255

            self.labels = data.iloc[:, 0]

        else:

            self.images = data / 255

            self.labels = np.empty(len(data))

            

    def __len__(self):

        return len(self.images)



    def __getitem__(self, idx):

        if isinstance(idx, int):

            idx = list([idx])



        images = torch.from_numpy(self.images.iloc[idx].values).float()

        

        labels = torch.from_numpy(np.array(self.labels[idx]))

        

        return images, labels



    def __iter__(self):

        for i in range(0, len(self), self.batch_size):

            yield self[i + np.arange(self.batch_size)]
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
## Read Digits Dataset

train_set = MNIST('../input/digit-recognizer/train.csv', batch_size=4)

test_set = MNIST('../input/digit-recognizer/test.csv', batch_size=1, train=False)



label_map = list(range(10))
## Read Fashion dataset

train_set = MNIST('/kaggle/input/fashionmnist/fashion-mnist_train.csv', batch_size=16)

test_set = MNIST('/kaggle/input/fashionmnist/fashion-mnist_test.csv', batch_size=1, train=True, shuffle=False)  ## Each test image has the expected label, train=True



label_map = {

    0: 'Top',

    1: 'Trouser',

    2: 'Pullover',

    3: 'Dress',

    4: 'Coat',

    5: 'Sandals',

    6: 'Shirt',

    7: 'Sneaker',

    8: 'Bag',

    9: 'Boot',

}
class Encoder(nn.Module):

    def __init__(self, layer_data):

        super(Encoder, self).__init__()



        incoming, hidden, out = layer_data



        self.la = nn.Linear(incoming, hidden)

        

        self.l_mu = nn.Linear(hidden, out)

        self.l_logvar = nn.Linear(hidden, out)

    

    def forward(self, x):

        """

        Forward pass through encoder and decoder.

        """

        x = F.relu(self.la(x))



        mu = self.l_mu(x)

        logvar = self.l_logvar(x)  ## logvar = log(sigma^2)



        return mu, logvar
class Decoder(nn.Module):

    def __init__(self, layer_data):

        super(Decoder, self).__init__()



        incoming, hidden, out = layer_data



        self.ly = nn.Linear(out, hidden)

        self.lz = nn.Linear(hidden, incoming)



    def forward(self, z):

        """

        Forward pass through encoder and decoder.

        """

        y = F.relu(self.ly(z))

        y = torch.sigmoid(self.lz(y))  # Sigmoid bounds image to [-1, 1]?



        return y
class VAE(nn.Module):

    def __init__(self, layer_data):

        super(VAE, self).__init__()

        

        self.encoder = Encoder(layer_data)

        self.decoder = Decoder(layer_data)

    

    def forward(self, x):

        """

        Forward pass through encoder and decoder.

        """

        mu, logvar = self.encoder.forward(x)



        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        

        z = mu + eps*std



        y = self.decoder.forward(z)



        return y, mu, logvar
## Initialize

vae = VAE([784, 512, 2])



def loss_function(recon_x, x, mu, logvar):

    """

    Cross entropy + KL Divergence loss

    """

    BCE = F.binary_cross_entropy(recon_x, x.view((-1, 784)), reduction='sum')



    # https://arxiv.org/abs/1312.6114

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())



    return BCE + KLD



optimizer = optim.Adam(vae.parameters(), lr=1e-3)
## Train

vae.train()



for epoch in range(3):

    optimizer.zero_grad()

    

    total_loss = 0

    for i, data in enumerate(train_set):

        inputs, labels = data



        optimizer.zero_grad()



        outputs, mu, logvar = vae(inputs)

        

        loss = loss_function(outputs, inputs, mu, logvar)

        loss.backward()

        

        total_loss += loss.item()



        optimizer.step()



        total_loss += loss.item()

        if i % 2000 == 1999:

            print(f'{epoch+1}, {i+1} - loss: {total_loss / 2000:.3f}')

            total_loss = 0.0
## Generate

vae.eval()



# predictions = [vae(inputs)[0].data for inputs, labels in test_set]
label_map
## Show off

offset = 0



for idx in range(10):

    image, labels = test_set[idx]



    while label_map[int(labels[0])] != 'Boot':        

        offset += 1

        

        image, labels = test_set[idx + offset]

    

    plt.imshow(vae(image)[0].data.reshape((28, 28)))

    plt.show()
