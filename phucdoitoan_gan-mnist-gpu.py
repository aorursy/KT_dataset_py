# architecture reference: 

# https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN

# https://github.com/lyeoni/pytorch-mnist-GAN



# implementing Vanilla GAN on MNIST



# 1. libraries loading

# 2. data prepare + batch_size + num of iteration + epochs

# 3. defining Generator and Discriminator models 

# 4. instantiate Generator and Discriminator + loss function 

#    + learning_rate + optimizers

# 5. training D and G

# 6. visualize samples generated by G
# 1. libraries loading



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import torch

import torch.nn as nn

import torch.utils.data

from torch.autograd import Variable

from sklearn.model_selection import train_test_split

from torchvision.utils import save_image

from timeit import default_timer as timer

import time

# 2. data loading + batch_size + epochs + batch_num



# gpu or cpu

dtype = torch.FloatTensor

dtype = torch.cuda.FloatTensor # uncomment this to run on GPU



# loading data + normalize pixels' values between -1 and +1

data = pd.read_csv('../input/mnist_train.csv')

#labels = data.label.to_numpy()

img_digits = (data.loc[:, data.columns != 'label'].values/255 - 0.5)*2



# batch_size + iteration num

batch_size = 128

epochs_num = 200



# change DataFrame to numpy

digits_Tensor = torch.Tensor(img_digits).type(dtype)



# build Dataset

digits_DataSet = torch.utils.data.TensorDataset(digits_Tensor)



# build DataLoader

digits_DataLoader = torch.utils.data.DataLoader(digits_DataSet, batch_size = batch_size)



batch_num = len(digits_DataLoader) # numbers of mini batches



# visualize data

first_four = np.append(np.append(img_digits[0].reshape(28,28), img_digits[1].reshape(28,28), axis=0), np.append(img_digits[2].reshape(28,28), img_digits[3].reshape(28,28), axis = 0), axis=1)

plt.imshow(first_four)
# define functions to return mini batch of real data, noise

# sequentially return mini batch of real data from DataLoader after each call

def sample_real():

    while True:

        for i, [batch] in enumerate(digits_DataLoader):

            yield batch

sample_real = sample_real() # sample_real now is a generator



# randomly return a batch of noise

def sample_noise(size=batch_size):

    batch = torch.from_numpy(np.random.randn(size,100)).type(dtype)

    return batch
# 3. defining Generator and Discriminator



# generator: 100 -> 256 -> 512 -> 1024 -> 784

class Generator(nn.Module):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):

        super(Generator, self).__init__()

        

        # 1st layer

        self.fc1 = nn.Linear(input_dim, hidden_dim1)

        self.ac1 = nn.LeakyReLU(negative_slope=0.2)

        # 2nd layer

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)

        self.ac2 = nn.LeakyReLU(negative_slope=0.2)

        # 3rd layer

        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)

        self.ac3 = nn.LeakyReLU(negative_slope=0.2)

        # 4th layer (readout)

        self.fc4 = nn.Linear(hidden_dim3, output_dim)

        self.ac4 = nn.Tanh()

        

    def forward(self, x):

        out1 = self.ac1(self.fc1(x))

        out2 = self.ac2(self.fc2(out1))

        out3 = self.ac3(self.fc3(out2))

        out = self.ac4(self.fc4(out3))

        

        return out

    

# discriminator: 784 -> 1024 -> 512 -> 256 -> 1

class Discriminator(nn.Module):

    def __init__(self,input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):

        super(Discriminator, self).__init__()

        

        # 1st layer

        self.fc1 = nn.Linear(input_dim, hidden_dim1)

        self.ac1 = nn.LeakyReLU(negative_slope=0.2)

        self.dropout1 = nn.Dropout(p=0.3)

        # 2nd layer

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)

        self.ac2 = nn.LeakyReLU(negative_slope=0.2)

        self.dropout2 = nn.Dropout(p=0.3)

        # 3rd layer

        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)

        self.ac3 = nn.LeakyReLU(negative_slope=0.2)

        self.dropout3 = nn.Dropout(p=0.3)

        # 4th layer (readout)

        self.fc4 = nn.Linear(hidden_dim3, output_dim)

        self.ac4 = nn.Sigmoid()

        

    def forward(self, x):

        out1 = self.dropout1(self.ac1(self.fc1(x)))

        out2 = self.dropout2(self.ac2(self.fc2(out1)))

        out3 = self.dropout3(self.ac3(self.fc3(out2)))

        out = self.ac4(self.fc4(out3))

        

        return out

        
# 4. instantiate G and D; loss function; learning_rate; optimizers



# generator + discriminator

G = Generator(100, 256, 512, 1024, 784).type(dtype)

D = Discriminator(784, 1024, 512, 256, 1).type(dtype)



# loss function: Binary Cross Entropy Loss

error = nn.BCELoss()



# learning rate

learning_rate = 0.0002



# optimizers: 

G_optimizer = torch.optim.Adam(G.parameters(), lr = learning_rate)

D_optimizer = torch.optim.Adam(D.parameters(), lr = learning_rate)

# visualize image generated by generator before training

before_noise = sample_noise(1)

before_result = G(before_noise).cpu() # copy tensor to host memory



before_img = before_result.detach().numpy().reshape(28,28)

plt.imshow(before_img)
# 5. training D and G



# list to store loss, iter

G_loss_list = []

D_real_loss_list = []

D_fake_loss_list = []

iter_list = []



# function to save digit images generated by G after each epoch

def G_generate(epoch, size=batch_size):

    generated = G(sample_noise(size)).cpu() #copy tensor to host

    save_image(generated.view(generated.size(0), 1, 28, 28), 'epoch' + str(epoch) + '.png')



all_start = timer()

for epoch in range(epochs_num):

    epoch_start = timer()

    for it, [real_batch] in enumerate(digits_DataLoader):

        

        ##### TRAIN D  #####

        # on real batch

        # clear gradients

        D_optimizer.zero_grad()

        # forward propagation

        real_out = D(real_batch)

        # soft label for real batch: 1

        real_label = torch.ones(real_out.shape[0], 1).type(dtype)

        # loss

        D_real_loss = error(real_out, real_label)

        # back propagation

        D_real_loss.backward()

        # update parameters

        D_optimizer.step()

    

        # on fake batch

        # sample noise + generate fake batch

        noise_batch = sample_noise()

        fake_batch = G(noise_batch)

        # clear gradients

        D_optimizer.zero_grad()

        # forward propagation

        fake_out = D(fake_batch)

        # soft label for fake batch: 0

        fake_label = torch.zeros(fake_out.shape[0], 1).type(dtype)

        # loss

        D_fake_loss = error(fake_out, fake_label)

        # back propagation

        D_fake_loss.backward()

        # update parameters

        D_optimizer.step()        

        

        ##### TRAIN G #####

        # sample noise + generate fake batch

        noise_batch = sample_noise()

        fake_batch = G(noise_batch)

        # clear gradients

        G_optimizer.zero_grad()

        # forward propagation

        fake_out = D(fake_batch)

        # soft label for fake batch: 1

        fake_label = torch.ones(fake_out.shape[0], 1).type(dtype)

        # loss 

        G_loss = error(fake_out, fake_label)

        # back propagation

        G_loss.backward()

        # update parameters

        G_optimizer.step()

        

        ##### store loss, it #####

        if it % 500 == 0:

            iter_list.append(it+batch_num*epoch)

            G_loss_list.append(G_loss.data)

            D_fake_loss_list.append(D_fake_loss.data)

            D_real_loss_list.append(D_real_loss.data)

                    

            #print('Epoch {} [{}/{}]:'.format(epoch, it, batch_num))



    print('Epoch {} :'.format(epoch))  

    print('    G_loss: {}'.format(G_loss.data))

    print('    D_fake_loss: {}'.format(D_fake_loss.data))

    print('    D_real_loss: {}'.format(D_real_loss.data))

            

    epoch_duration = timer() - epoch_start

    G_generate(epoch)

    print('##### TRAIN EPOCH {} IN {} s #####'.format(epoch, epoch_duration))

    

all_duration = timer() - all_start

print('ALL TRAINING TIME: {}'.format(all_duration))
# plot D and G loss

plt.plot(iter_list, G_loss_list, 'r')

plt.plot(iter_list, D_real_loss_list, 'b')

plt.plot(iter_list, D_fake_loss_list, 'g')
# 6. visualize samples generated by G

generated = G(sample_noise()).cpu()

img_test = generated[0].detach().numpy().reshape(28,28)

plt.imshow(img_test)