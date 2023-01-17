import numpy as np 

import pandas as pd 

import os

import cv2

import matplotlib.pyplot as plt

%matplotlib inline
import torch

from torchvision import datasets, transforms, models

from torch.utils.data.sampler import SubsetRandomSampler
directory = '/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images'

classes = ['Parasitized','Uninfected']
transform = transforms.Compose([transforms.Resize(255),

                                transforms.CenterCrop(224),

                                transforms.ToTensor(),

                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])



data = datasets.ImageFolder(directory,transform=transform)



batch_size = 64

valid_size = 0.15

test_size = 0.20

num_train = len(data)

indices = list(range(num_train))

np.random.shuffle(indices)

# spliting into test and train dataset

split1 = int(np.floor(test_size * num_train))

train_idx, test_idx = indices[split1:], indices[:split1]

# spliting into validation and train dataset

indices = list(range(num_train-split1))

split2 = int(np.floor(valid_size * (num_train-split1)))

train_idx, valid_idx = indices[split2:], indices[:split2]



# define samplers for obtaining training and validation batches

train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)

test_sampler = SubsetRandomSampler(test_idx)



# prepare data loaders (combine dataset and sampler)

train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,sampler=valid_sampler)

test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=test_sampler)
print("Train dataset size = ",len(train_loader)*batch_size*100/num_train)

print("Validation dataset size = ",len(valid_loader)*batch_size*100/num_train)

print("Test dataset size = ",len(test_loader)*batch_size*100/num_train)
from torch import nn

from torch import optim



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = models.densenet121(pretrained=True)



# Freeze parameters so we don't backprop through them

for param in model.parameters():

    param.requires_grad = False

    

model.classifier = nn.Sequential(nn.Linear(1024, 256),

                                 nn.ReLU(),

                                 nn.Dropout(0.2),

                                 nn.Linear(256, 2),

                                 nn.LogSoftmax(dim=1))



criterion = nn.NLLLoss()



# Only train the classifier parameters, feature parameters are frozen

optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)



model.to(device)
epochs = 15

steps = 0

running_loss = 0

print_every = 5

for epoch in range(epochs):

    for inputs, labels in train_loader:

        steps += 1

        # Move input and label tensors to the default device

        inputs, labels = inputs.to(device), labels.to(device)

        

        optimizer.zero_grad()

        

        logps = model.forward(inputs)

        loss = criterion(logps, labels)

        loss.backward()

        optimizer.step()



        running_loss += loss.item()

        

        if steps % print_every == 0:

            test_loss = 0

            accuracy = 0

            model.eval()

            with torch.no_grad():

                for inputs, labels in test_loader:

                    inputs, labels = inputs.to(device), labels.to(device)

                    logps = model.forward(inputs)

                    batch_loss = criterion(logps, labels)

                    

                    test_loss += batch_loss.item()

                    

                    # Calculate accuracy

                    ps = torch.exp(logps)

                    top_p, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    

            print(f"Epoch {epoch+1}/{steps}.. "

                  f"Train loss: {running_loss/print_every:.3f}.. "

                  f"Test loss: {test_loss/len(test_loader):.3f}.. "

                  f"Test accuracy: {accuracy/len(test_loader):.3f}")

            running_loss = 0

            model.train()
# obtain one batch of test images

dataiter = iter(test_loader)

images, labels = dataiter.next()

images.numpy()



def imshow(img):

    img = img / 2 + 0.5  # unnormalize

    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image



# move model inputs to cuda, if GPU available

if torch.cuda.is_available():

    images = images.cuda()



# get sample outputs

output = model.forward(images)

# convert output probabilities to predicted class

_, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(preds_tensor.cpu().numpy())



# plot the images in the batch, along with predicted and true labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    imshow(images.cpu()[idx])

    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),

                 color=("green" if preds[idx]==labels[idx].item() else "red"))