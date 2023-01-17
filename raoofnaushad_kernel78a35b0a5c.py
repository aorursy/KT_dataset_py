# !pip install matplotlib

import torch

import torchvision

from torchvision.datasets import MNIST

import matplotlib.pyplot as plt

import torchvision.transforms as transforms

from torch.utils.data import random_split, DataLoader

import torch.nn.functional as F

%matplotlib inline
dataset = MNIST(root='data/', download=True)

test = MNIST(root='data/', train=False)

### Checking the dataset



image, label = dataset[0]

plt.imshow(image, cmap='gray')

print("Label", label)
dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())

## Checking Dataset shape

image, label = dataset[0]

print(image.shape, label," ## 1 Channel length, 28*28 pixel image")



## Splitting the dataset

train, valid = random_split(dataset, [50000, 10000])



## Dataloader for loading train and valid data

batch_size = 128

trainLoader = DataLoader(train, batch_size, shuffle=True)

validLoader = DataLoader(valid, batch_size)
## Losses with Cross Entropy

loss_fn = F.cross_entropy



def accuracy(outputs, labels):

    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds==labels).item()/ len(preds))
import torch.nn as nn



input_size = 28*28

classes = 10



class MnistModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.linear = nn.Linear(input_size, classes)

    def forward(self, xb):

        xb = xb.reshape(-1, 28*28)

        out = self.linear(xb)

        return out

    def training_step(self, batch):

        images, labels = batch

        outputs = self(images) 

        loss = loss_fn(outputs, labels)

        return loss

    def valid_step(self, batch):

        images, labels = batch

        outputs = self(images)

        loss = loss_fn(outputs, labels)

        acc = accuracy(outputs, labels)

        return {"val_loss":loss, "val_acc":acc}

    def valid_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses

        batch_accs = [x['val_acc'] for x in outputs]

        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    

    def epoch_end(self, epoch, result):

        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

    

model = MnistModel()

## Some Stats of the model

print(model.linear.weight.shape, model.linear.bias.shape)

print(list(model.parameters()))

## Evaluation

def evaluate_model(model, valid_loader):

    result = [model.valid_step(batch) for batch in valid_loader]

    return model.valid_epoch_end(result)



history = []

### Fitting data

def fit(epochs, model,train_loader, valid_loader, lr, opt_func = torch.optim.SGD):

    historry = []

    ## Defining optimizer

    optimizer = opt_func(model.parameters(), lr)

    for i in range(epochs):

        ## Training

        for batch in train_loader:

            loss = model.training_step(batch)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            

        result = evaluate_model(model, valid_loader)

        model.epoch_end(epochs, result)

        history.append(result)

    return history

        
## Fitting

history1 = fit(5, model, trainLoader, validLoader, 0.001)
### Plotting Graph

history = history1 #+ history2 + history3 + history4

accuracies = [result['val_acc'] for result in history]

plt.plot(accuracies, '-x')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.title('Accuracy vs. No. of epochs')