

%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import torchvision

import time

from torchvision import transforms,models,datasets

import torch

import numpy as np

import matplotlib.pyplot as plt

import torch.optim as optim

import torch.nn as nn

from collections import OrderedDict

from PIL import Image

import numpy as np 

import pandas as pd 

import os

import json

print(os.listdir("../input"))





train_transforms=transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.CenterCrop(224),

#         transforms.RandomRotation(45),

#         transforms.RandomResizedCrop(224),

#         transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], 

                             [0.229, 0.224, 0.225])

    ])

valid_transforms=transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], 

                             [0.229, 0.224, 0.225])

    ])

train_datasets = datasets.ImageFolder('../input/flower_data/flower_data/train',transform=train_transforms)

valid_datasets = datasets.ImageFolder('../input/flower_data/flower_data/valid',transform=valid_transforms)

test_datasets = datasets.ImageFolder('../input/test set/',transform=valid_transforms)



trainloader=torch.utils.data.DataLoader(train_datasets, batch_size=512, shuffle=True)

validloader=torch.utils.data.DataLoader(valid_datasets, batch_size=512, shuffle=True)

testloader=torch.utils.data.DataLoader(test_datasets, batch_size=512)

print("training examples : ",len(trainloader.dataset))

print("validation examples : ",len(validloader.dataset))

print("test examples : ",len(testloader.dataset))



print("training batches: ",len(trainloader))

print("validation batches : ",len(validloader))

print("test batches : ",len(testloader))



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

print(images[0].shape)
model = models.densenet121(pretrained=True) # we will use a pretrained model and we are going to change only the last layer

for param in model.parameters():

  param.requires_grad= False

classifier  = nn.Sequential(nn.Linear(1024, 102),

                      nn.LogSoftmax(dim=1))

model.classifier=classifier

if torch.cuda.is_available():

  model.to('cuda')

  device='cuda'

else:

  model.to('cpu')

  device='cpu'

print(device)

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.003,weight_decay=0.0001)

valid_loss_min = 99 #just a big number I could do np.Inf

save_file='mymodel.pth'

epochs = 30

train_loss_array = []

test_loss_array  = []

running_loss = 0

for epoch in range(epochs):

    time0=time.time()

    for inputs, labels in trainloader:

        # Move input and label tensors to the default device

        inputs, labels = inputs.to(device), labels.to(device)

        

        optimizer.zero_grad()

        

        logps = model.forward(inputs)

        loss = criterion(logps, labels)



        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        

    else:

        valid_loss = 0

        accuracy = 0

        model.eval()

        with torch.no_grad():

            for inputs, labels in validloader:

                inputs, labels = inputs.to(device), labels.to(device)

                logps = model.forward(inputs)

                batch_loss = criterion(logps, labels)

                valid_loss += batch_loss.item()



                    # Calculate accuracy

                ps = torch.exp(logps)

                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            total_loss=valid_loss/len(validloader)        

            print(f"Epoch {epoch+1}/{epochs}.. "

            f"Train loss: {running_loss/(len(trainloader)*2):.3f}.. "

            f"valid loss: {valid_loss/len(validloader):.3f}.. "

            f"valid accuracy: {accuracy/len(validloader):.3f}")

            time_total=time.time() - time0

            print("time for this epoch: ",end="")

            print(time_total)

            train_loss_array.append(running_loss/(len(trainloader)*2))

            test_loss_array.append(valid_loss/len(validloader))

            if (total_loss) <= valid_loss_min:

                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,total_loss))

                torch.save(model.state_dict(), save_file)

                valid_loss_min = total_loss

            running_loss = 0

        model.train()

print(time)
plt.plot(train_loss_array)

plt.plot(test_loss_array)

plt.show()
device = "cpu"

model.to(device)

model.eval()

with torch.no_grad():

    for inputs, labels in validloader:

        inputs, labels = inputs.to(device), labels.to(device)

        logps = model.forward(inputs)

        ps = torch.exp(logps)

        top_p, top_class = ps.topk(1, dim=1)

print(logps)

with open('../input/cat_to_name.json', 'r') as f:

    cat_to_name = json.load(f)
top_class = top_class.numpy().reshape(-1)

predicted_results = pd.DataFrame(columns=['output'])

print(top_class.shape[0])

i = 0

for e in top_class:

    predicted_results.loc[i] = cat_to_name[str(e+1)]

    i += 1



predicted_results.head()