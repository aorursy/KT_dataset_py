#load data

import torch

import torch.optim as optim

import torch.nn.functional as F

import matplotlib.pyplot as plt



# general imports

import numpy as np

import time

import os

import glob

import tarfile



import utils

import dataa as D

#import controller as C

import child_model as CM

#import child_visualizer as V

import time



# set RNG seed for reproducibility

seed = 42

torch.manual_seed(seed)

np.random.seed(seed)
def save_checkpoint(experiment_name, enas_epoch, model, weights, accs, losses, train_losses, path="saved"):

    full_path = path + P + experiment_name

    if not os.path.exists(full_path):

        os.makedirs(full_path)

    

    g = time.gmtime()

    check_name  = experiment_name+"_"+str(g.tm_year)+"_"+str(g.tm_mon)+"_"+str(g.tm_mday)+"_"+str(g.tm_hour)+"_"+str(g.tm_min)+"_"+str(g.tm_sec)+".pt"

    torch.save({"epoch": enas_epoch,

                "model": model,

                "shared_weights": weights,

                "accs": accs,

                "losses": losses,

                "train_losses": train_losses

                },

                full_path+P+check_name)
# extract CIFAR10 dataset

tar = tarfile.open("../input/cifar10-python/cifar-10-python.tar.gz", "r:gz")

tar.extractall()

tar.close()

P = os.path.sep
# load the dataset

BATCH_SIZE = 100

TEST_BATCH_SIZE = 200

cifar10 = D.CIFAR10(batch_size=BATCH_SIZE, path="./", test_batch_size=TEST_BATCH_SIZE)

test_set = list(iter(cifar10.test))

train_set = list(iter(cifar10.train))
ops = [3,1,1,3,4,2,1,2,3,4,1,3,0,3]

#no avgpooling: 4 convs, one maxpool, (like human design), a lot of 5x5 -> low network depth, a lot sep -> sep conv faster since less params?

#lots of skips -> might help



skips = [0, 1,0, 1,0,0, 0,0,0,0, 0,0,0,1,0, 0,0,1,1,0,1, 1,0,1,1,0,1,1, 1,0,0,1,0,1,0,1, 0,0,0,0,0,0,1,0,0,

         1,1,0,0,0,0,0,1,0,0, 0,1,1,0,0,1,0,0,0,0,1, 1,0,1,1,0,1,0,1,0,0,0,1, 0,0,0,1,0,0,0,1,0,0,0,1,0,]



best_model = CM.ChildModel(ops, skips)

best_model = best_model.to_torch_model(channels = 24)
experiment = 'original_best_model24_lowlr'

checkpoint_interval = 52

model = best_model

optimizer = optim.Adam(

    model.parameters(), lr = 0.01, weight_decay= 1e-4

) #ADAM standard lr = 0.001

#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min = 0.001)

loss_func = F.cross_entropy



epochs = 105

log_interval = 100

accs =[]

losses=[]

train_losses=[]



# initial test

model.eval()

test_loss = 0

correct = 0

test_loader = test_set

testiter = test_loader

size_dataset = 10000 #correct?

with torch.no_grad():

    for data, target in testiter:

        output = model(data)

        test_loss += loss_func(output, target, reduction='sum').item() # sum up batch loss

        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= size_dataset

    acc = correct/size_dataset

print("Accuracy: ", acc)

print("Loss: ", test_loss)

accs.append(acc)

losses.append(test_loss)

train_losses.append(float('nan'))



    

batch_size = BATCH_SIZE

num_batches = len(train_set)

train_size = num_batches*batch_size

t1 = time.time()



for epoch in range(epochs):

    epoch += 1

    print('Epoch ', epoch, ' / ', epochs )



    for batch_idx, (data, target) in enumerate(train_set):



        optimizer.zero_grad()

        model.train()

        output = model(data)



        loss = loss_func(output, target)



        loss.backward()

        optimizer.step()



        if batch_idx % log_interval == 0: 

            print('Train [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Time: {:.3f}'.format(batch_idx*batch_size, train_size,

                100*batch_idx/num_batches, loss.item(), time.time() - t1))

    t1 = time.time()

    train_losses.append(loss.item())

    # test

    model.eval()

    test_loss = 0

    correct = 0

    test_loader = test_set

    testiter = test_loader

    size_dataset = 10000 #correct?

    with torch.no_grad():

        for data, target in testiter:

            output = model(data)

            test_loss += loss_func(output, target, reduction='sum').item() # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= size_dataset

        acc = correct/size_dataset

    print("Accuracy: ", acc)

    print("Loss: ", test_loss)

    #scheduler.step()

    accs.append(acc)

    losses.append(test_loss)

    if epoch % checkpoint_interval == 0:

        save_checkpoint(experiment, epoch, model, model.state_dict(), accs, losses, train_losses)
