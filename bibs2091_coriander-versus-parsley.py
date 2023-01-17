%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import torchvision

import gc



import time

from torchvision import transforms,models,datasets

import torch

import numpy as np

import matplotlib.pyplot as plt

import torch.optim as optim

import torch.nn as nn

from collections import OrderedDict

from PIL import Image

import seaborn as sns

import helper

import numpy as np 

import pandas as pd 

import json

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

print(os.listdir("../input/coriandr-vs-parsley/Coriander-vs-Parsley-master"))
train_transforms=transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], 

                             [0.229, 0.224, 0.225])

    ])

test_transforms=transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], 

                             [0.229, 0.224, 0.225])

    ])

train_datasets = datasets.ImageFolder('../input/coriandr-vs-parsley/Coriander-vs-Parsley-master/train',transform=train_transforms)

test_datasets = datasets.ImageFolder('../input/coriandr-vs-parsley/Coriander-vs-Parsley-master/test',transform=test_transforms)



trainloader=torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)

testloader=torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)

print("train size: " , len(trainloader.dataset))

print("test size: " , len(testloader.dataset))

def imshow(image, ax=None, title=None, normalize=True):

  """Imshow for Tensor."""

  if ax is None:

      fig, ax = plt.subplots()

  image = image.numpy().transpose((1, 2, 0))



  if normalize:

      mean = np.array([0.485, 0.456, 0.406])

      std = np.array([0.229, 0.224, 0.225])

      image = std * image + mean

      image = np.clip(image, 0, 1)



  ax.imshow(image)

  ax.spines['top'].set_visible(False)

  ax.spines['right'].set_visible(False)

  ax.spines['left'].set_visible(False)

  ax.spines['bottom'].set_visible(False)

  ax.tick_params(axis='both', length=0)

  ax.set_xticklabels('')

  ax.set_yticklabels('')



  return ax

data_iter = iter(trainloader)

images, labels = next(data_iter)

imshow(images[0])

print(labels[0])

# we will use a pretrained model and we are going to change only the last layer 

model = models.densenet201(pretrained=True)

for param in model.parameters():

  param.requires_grad= True
print(model)
classifier  = nn.Sequential(nn.Linear(1920, 256),

                          nn.ReLU(),

                          nn.Linear(256, 2),

                          nn.LogSoftmax(dim = 1))

model.classifier=classifier

if torch.cuda.is_available():

  model.to('cuda')

  device='cuda'

else:

    model.to('cpu')

    device='cpu'

print(device)

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.00001,weight_decay=0)

test_loss_min = 99 #just a big number I could do np.Inf

save_file='mymodel.pth'
epochs = 200

train_losses = [] 

test_losses = []

print_every = 10

running_loss = 0

for epoch in range(epochs):

    time0=time.time()

    model.train()

    for inputs, labels in trainloader:

        # Move inp  ut and label tensors to the default device

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        prediction = model.forward(inputs)

        loss = criterion(prediction, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    

    else:

        train_losses.append(running_loss/(len(trainloader)))

        running_loss = 0

        if ((epoch % print_every) == 0):

            test_loss = 0

            accuracy = 0

            model.eval()

            with torch.no_grad():

                for inputs, labels in testloader:

                    inputs, labels = inputs.to(device), labels.to(device)

                    logps = model.forward(inputs)

                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()



                        # Calculate accuracy

                    ps = torch.exp(logps)

                    top_p, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                total_loss=test_loss/len(testloader)        

                print(f"Epoch {epoch+1}/{epochs}.. "

                f"Train loss: {running_loss/(len(trainloader) * print_every):.3f}.. "

                f"test loss: {test_loss/len(testloader):.3f}.. "

                f"test accuracy: {accuracy/len(testloader):.3f}")

                time_total=time.time() - time0

                print("time for this epoch: ",end="")

                print(time_total)

#                 train_losses.append(running_loss/(len(trainloader) * print_every))

                test_losses.append(total_loss)

                if (total_loss) <= test_loss_min:

                    print('test loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(test_loss_min,total_loss))

                    torch.save(model.state_dict(), save_file)

                    test_loss_min = total_loss

                running_loss = 0

plt.plot(train_losses)

# plt.plot([k for k in range(0,epochs,print_every)],test_losses)

plt.show()
del model

del testloader

del trainloader

del inputs

del labels

torch.cuda.empty_cache()