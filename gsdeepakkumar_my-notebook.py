# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from torchvision import datasets

import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt

from torchvision import models

from torch import nn

import torch

from torch import optim

import torch.nn.functional as F

from tqdm import tqdm
# check if CUDA is available

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
train_transform=transforms.Compose([

    transforms.Resize(224),

    transforms.RandomRotation(0.3),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])

test_transform=transforms.Compose([

    transforms.Resize(224),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])





train_data =  datasets.ImageFolder('../input/building-type/staticmap/train', transform=train_transform)

test_data =  datasets.ImageFolder('../input/building-type/staticmap/test', transform=test_transform)

train_data
# number of subprocesses to use for data loading

num_workers = 0

# how many samples per batch to load

batch_size = 16

# percentage of training set to use as validation

valid_size = 0.20



# obtain training indices that will be used for validation

num_train = len(train_data)

indices = list(range(num_train))

np.random.shuffle(indices)

split = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]



train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,

    sampler=train_sampler, num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 

    sampler=valid_sampler, num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 

    num_workers=num_workers)
classes = ['residential','industrial']

def imshow(img):

    img = img / 2 + 0.5  # unnormalize

    plt.imshow(np.transpose(img, (1, 2, 0)))

dataiter = iter(train_loader)

images, labels = dataiter.next()

images = images.numpy() # convert images to numpy for display



# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))

# display 20 images

for idx in np.arange(16):

    ax = fig.add_subplot(2, 16/2, idx+1, xticks=[], yticks=[])

    imshow(images[idx])

    ax.set_title(classes[labels[idx]])
model=models.alexnet(pretrained=True)
model.classifier[6].out_features=2
model
## Freeze the parameters so that we dont require to backpropagate them:

for param in model.features.parameters():

    param.requires_grad = False

    

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model=model.to(device)
epochs = 25

running_loss = 0

train_losses, valid_losses, accuracy_total = [], [], []



steps = 0

print_every = 50



for epoch in tqdm(range(epochs)):



    running_loss = 0

    valid_loss = 0

    accuracy = 0

    

    model.train()    

    for inputs, labels in (train_loader):

        #steps += 1

        

        # Move input and label tensors to the default device

        inputs, labels = inputs.to(device), labels.to(device)

        

        # clear the gradients of all optimized variables 

        optimizer.zero_grad()

        # forward pass: compute predicted outputs

        log_ps = model.forward(inputs)

        # batch loss

        loss = criterion(log_ps, labels)

        # backward pass

        loss.backward()        

        optimizer.step()



        running_loss += loss.item()

        

    # Validation Pass

    model.eval()                

    

    # validation pass

    for inputs, labels in valid_loader:

        inputs, labels = inputs.to(device), labels.to(device)

        #Log Softmax

        logps = model.forward(inputs)

        valid_loss = criterion(logps, labels)

        

        # Calculate accuracy

        ps = torch.exp(logps)

        top_p, top_class = ps.topk(1, dim=1)

        equals = top_class == labels.view(*top_class.shape)

        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()



        # check denominator for accuracy calcn here and in the print statement



    train_losses.append(running_loss/len(train_loader.sampler))

    valid_losses.append(valid_loss/len(valid_loader.sampler))

    accuracy_total.append((accuracy/len(valid_loader.sampler))*100)

                  

    #if steps % print_every == 0:

    print(f"Epoch {epoch+1}/{int(steps/print_every)}.. "

          f"Train loss: {running_loss/len(train_loader):.3f}.. "

          f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "

          f"Test accuracy: {(accuracy/len(valid_loader))*100:.3f}")

           
plt.plot(train_losses, label='Training Loss')

plt.plot(valid_losses, label='Validation Loss')

plt.legend(frameon=False)
plt.plot(valid_losses, label='Validation Loss', color='orange')

plt.legend(frameon=False)
# track test loss

train_on_gpu = torch.cuda.is_available()



test_loss = 0.0

class_correct = list(0. for i in range(2))

class_total = list(0. for i in range(2))



model.eval()

# iterate over test data

for data, target in test_loader:

    # move tensors to GPU if CUDA is available

    if train_on_gpu:

        data, target = data.cuda(), target.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data)

    # calculate the batch loss

    loss = criterion(output, target)

#     print(f'Data Size:{data.size(0)}')

#     print(f'Test dataset length {len(test_loader.dataset)}')

#     print(f"\n Trial : {loss.item()}")

#     print(f"\n Trial : {loss.item()*data.size(0)}")

    # update test loss 

    test_loss += loss.item()*data.size(0)  ## Taking the weighted sum

    # convert output probabilities to predicted class

    _, pred = torch.max(output, 1)    

    # compare predictions to true label

    correct_tensor = pred.eq(target.data.view_as(pred))

    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

    # calculate test accuracy for each object class



    for i in range(len(target)):

        label = target.data[i]

        class_correct[label] += correct[i].item()

        class_total[label] += 1



# average test loss

test_loss = test_loss/len(test_loader.dataset)

print('Test Loss: {:.6f}\n'.format(test_loss))



for i in range(2):

    if class_total[i] > 0:

        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (

            classes[i], 100 * class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:

        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))



print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))
## Predict on test set:

dataiter = iter(test_loader)

images, labels = dataiter.next()

images.numpy()



# move model inputs to cuda, if GPU available

if train_on_gpu:

    images = images.cuda()



# get sample outputs

output = model(images)

# convert output probabilities to predicted class

_, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())



# plot the images in the batch, along with predicted and true labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(16):

    ax = fig.add_subplot(2, 16/2, idx+1, xticks=[], yticks=[])

    imshow(images.cpu()[idx])

    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),

                 color=("green" if preds[idx]==labels[idx].item() else "red"))