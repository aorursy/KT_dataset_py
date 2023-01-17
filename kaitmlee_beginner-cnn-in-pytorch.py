# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np

import pandas as pd

import torch

from torch import nn, optim

import torch.nn.functional as F

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import random

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
X_train = train_data.values

X_train_all = X_train[:, 1:]

y_train_all = train_data.label.to_numpy()



X_test, X_train = np.split(X_train_all, [5000], axis=0)

y_test, y_train = np.split(y_train_all, [5000], axis=0)



X_real_test = test_data.values
print("There are %s digits for training with %s pixels each" %(X_train.shape[0], X_train.shape[1]))

print("There are %s digits for testing" %(X_test.shape[0]))

print("There are %s digits for the final test" %(X_real_test.shape[0]))
fig, axs = plt.subplots(figsize=(7, 7), nrows=2, ncols=5)

for row in range(0,2):

    for col in range(0,5):

        rand_int = random.randint(0, 36999)

        rand_digit = X_train[rand_int].reshape((28,28))

        pos = axs[row][col].imshow(rand_digit, cmap="binary")

        axs[row][col].set_title("Label: %s" %(y_train[rand_int]))

plt.tight_layout()
fig, ax = plt.subplots(figsize=(10, 5), ncols=2)

counts1 = np.bincount(y_test)

ax[0].bar(range(10), counts1, width=0.8, align='center', color="bisque", edgecolor="black")

ax[0].set(xticks=range(10), xlim=[-1, 10])

ax[0].set_title('Distribution of testing digits')

ax[0].set_ylabel("count")

ax[0].set_xlabel("digits")



counts2 = np.bincount(y_train)

ax[1].bar(range(10), counts2, width=0.8, align='center', color="burlywood", edgecolor="black")

ax[1].set(xticks=range(10), xlim=[-1, 10])

ax[1].set_title('Distribution of training digits')

ax[1].set_xlabel("digits")

plt.show()
class Flatten(nn.Module):

    def forward(self, input):

        return input.view(input.size(0), -1)



    

class Net(nn.Module):

    def __init__(self, hls):

        super(Net, self).__init__()

        self.bn = nn.BatchNorm2d(1)

        

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        self.conv1_bn = nn.BatchNorm2d(10)

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_bn = nn.BatchNorm2d(20)        

#         self.conv2_drop = nn.Dropout2d()

        

        self.flatten = Flatten()

        self.fc1 = nn.Linear(320, hls)

        self.fc2 = nn.Linear(hls, 10)

        



    def forward(self, x):

#         x = F.relu(F.max_pool2d(self.conv1(x), 2))

#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = self.bn(x)

        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))

        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))     

#         x = self.conv2_drop(x)



        x = self.flatten(x)

        x = F.relu(self.fc1(x))

        x = F.dropout(x, training=self.training)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    

cnet = Net(310)

print("The network:")

print(cnet)
def score_cnn(net, criterion):

    net.eval()

    test_loss = 0

    correct = 0



    batch_sz = 1000

    N, D = X_test.shape

    n_batches = N // batch_sz

    for j in range(n_batches):

        Xbatch = X_test[j*batch_sz:(j*batch_sz+batch_sz)]

        Ybatch = y_test[j*batch_sz:(j*batch_sz+batch_sz)]

        Xbatch = Xbatch.reshape(Xbatch.shape[:-1] + (28,28))

        Xbatch = np.expand_dims(Xbatch, axis=1)



        data_test, target_test = torch.tensor(Xbatch), torch.tensor(Ybatch)

        data_test = data_test.type(torch.FloatTensor)

        net_out= net(data_test)

        # sum up batch loss

        target_test = target_test.type(torch.LongTensor)

        # get the index of the max log-probability (one hot encoding -> array of predictions)

        pred = net_out.data.max(1)[1]

        correct += pred.eq(target_test.data).sum()

    return (100. * correct.item() / X_test.shape[0])
def train(X_train, y_train, max_tries=30, epochs=40, hls=310, lr=12e-4):

    batch_sz = 1000

    N, D = X_train.shape

    print_period = 100



    best_validation_rate = 0

    best_hls = None

    best_lr = None

    best_losses = None

    best_cnet = None



    for _ in range(max_tries):

        cnet = Net(hls)

        # Adam worked better than the stochastic gradient descent optimizer

        optimizer = optim.Adam(cnet.parameters(), lr=lr)

        # Negative log likelihood loss

        criterion = nn.NLLLoss()

        cnet.train()



        n_batches = N // batch_sz

        losses = []

        for epoch in range(epochs):

            if n_batches > 1:

                X_train, y_train = shuffle(X_train, y_train)

            for j in range(n_batches):

                Xbatch = X_train[j*batch_sz:(j*batch_sz+batch_sz)]

                Ybatch = y_train[j*batch_sz:(j*batch_sz+batch_sz)]

                Xbatch = Xbatch.reshape(Xbatch.shape[:-1] + (28,28))

                Xbatch = np.expand_dims(Xbatch, axis=1)

                data = torch.from_numpy(Xbatch)

                target = torch.from_numpy(Ybatch)

                data, target=data.type(torch.DoubleTensor),target.type(torch.DoubleTensor)



                # initialize weights to zero

                optimizer.zero_grad()

                # log softmax output

                net_out = cnet(data.float())

                # negative log likelihood loss between the output of our network and our target batch data

                target = target.type(torch.LongTensor)

                loss = criterion(net_out, target)

                # gradients computed

                loss.backward()

                # gradient step

                optimizer.step()

                losses.append(loss.data.item())



                if j*batch_sz % 37000 == 0:

                    print('Train Epoch: {}  \tLoss: {:.6f}'.format(

                            epoch, loss.data.item()))



        validation_accuracy = score_cnn(cnet, criterion)

        print(

          "validation_accuracy: %.4f, settings: %s, %s" %

            (validation_accuracy, hls, lr)

        )

        print("")

        if validation_accuracy > best_validation_rate:

            best_validation_rate = validation_accuracy

            best_hls = hls

            best_lr = lr

            best_losses = losses

            best_cnet = cnet



        # select new hyperparams

        hls = best_hls + np.random.randint(-2, 4)*20

        lr = best_lr + np.random.randint(-2, 4)*1e-4



    print("Best validation_accuracy:", best_validation_rate)

    print("Best settings:")

    print("hidden layer size:", best_hls)

    print("learning_rate:", best_lr)

    

    return best_losses, best_cnet
best_losses, best_cnet = train(X_train=X_train, y_train=y_train, max_tries=5, epochs=30, hls=310, lr=11e-4)
plt.plot(best_losses)

plt.title("Final loss: %.3f" % best_losses[-1])

plt.xlabel("Iteration")

plt.ylabel("Loss")

plt.show()
# # losses = train(X_train=X_train_all, y_train=y_train_all, max_tries=1, epochs=50, hls=310, lr=11e-4)

# batch_sz = 1000

# N, D = X_train_all.shape

# print(X_train_all.shape)

# print_period = 100

# epochs = 30

# hls = 310

# lr = 12e-4



# cnet = Net(hls)

# # stochastic gradient descent optimizer

# #optimizer = optim.SGD(cnet.parameters(), lr=lr, momentum=m)

# optimizer = optim.Adam(cnet.parameters(), lr=lr)

# # Negative log likelihood loss

# criterion = nn.NLLLoss()

# cnet.train()



# n_batches = N // batch_sz

# for epoch in range(epochs):

#     if n_batches > 1:

#         X, Y = shuffle(X_train_all, y_train_all)

#     for j in range(n_batches):

#         Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]

#         Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

#         Xbatch = Xbatch.reshape(Xbatch.shape[:-1] + (28,28))

#         Xbatch = np.expand_dims(Xbatch, axis=1)

#         data = torch.from_numpy(Xbatch)

#         target = torch.from_numpy(Ybatch)

#         data, target=data.type(torch.DoubleTensor),target.type(torch.DoubleTensor)



#         # initialize weights to zero

#         optimizer.zero_grad()

#         # log softmax output

#         net_out = cnet(data.float())

#         # negative log likelihood loss between the output of our network and our target batch data

#         target = target.type(torch.LongTensor)

#         loss = criterion(net_out, target)

#         # gradients computed

#         loss.backward()

#         # gradient step

#         optimizer.step()



#         if j*batch_sz % 60000 == 0:

#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

#                     epoch, j * batch_sz, X_train_all.shape[0],

#                            100. * j * batch_sz / X_train_all.shape[0], loss.data.item()))
best_cnet.eval()

N, D = X_real_test.shape

X_real = X_real_test.reshape(X_real_test.shape[:-1] + (28,28))

X_real = np.expand_dims(X_real, axis=1)

data_test= torch.tensor(X_real)

data_test = data_test.type(torch.FloatTensor)

net_out = best_cnet(data_test)

pred = net_out.data.max(1)[1]

pred = pred.detach().numpy()
output = pd.DataFrame({'ImageId':range(1, 28001), 'Label':pred})

output
fig, axs = plt.subplots(figsize=(7, 7), nrows=5, ncols=5)

i = 0

for row in range(0,5):

    for col in range(0,5):

        rand_digit = X_real_test[i].reshape((28,28))

        pos = axs[row][col].imshow(rand_digit, cmap="binary")

        axs[row][col].set_title('Digit: %s' % output.Label[i])

        i = i+1

plt.tight_layout()
output.to_csv('submission.csv', index=False)