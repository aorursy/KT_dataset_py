import copy

import numpy as np

import matplotlib.pyplot as plt



import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler



import torchvision

from torchvision import datasets, transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([

    transforms.RandomResizedCrop(224),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(mean=mean, std = std)

])
test_transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=mean, std = std)

])
train_dir = '../input/spill-data/Spill_Data/Train/'

test_dir = '../input/spill-data/Spill_Data/Test/'



batch_size = 8

num_workers = 4
train_data = datasets.ImageFolder(root=train_dir,

                                 transform=train_transform)
train_data
train_loader = torch.utils.data.DataLoader(train_data,

                                          batch_size=batch_size,

                                          shuffle=True,

                                          num_workers=num_workers)
test_data = datasets.ImageFolder(root=test_dir,

                                 transform=test_transform)



test_data
test_loader = torch.utils.data.DataLoader(test_data,

                                          batch_size=batch_size,

                                          shuffle=True,

                                          num_workers=num_workers)
dataloaders = {

    'train': train_loader,

    'test': test_loader

}
total_batch_sizes = {'train': len(train_loader), 'test': len(test_loader)}



total_batch_sizes
class_names = train_data.classes

class_names
def imshow(inp, title):

    

    inp = inp.cpu().numpy().transpose((1,2,0))

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    

    plt.figure(figsize = (12,6))

    

    plt.imshow(inp)

    plt.title(title)

    plt.pause(5)
inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])
from torchvision import models
model = models.resnet18(pretrained=True)
model
num_ftrs = model.fc.in_features

num_ftrs
count = 0



for child in model.children():

    count+=1

    

count
count = 0



for child in model.children():

    count+=1

    if count < 7:

        for params in child.parameters():

            params.requires_grad = False
model.fc = nn.Linear(num_ftrs, 2)
model
criterion = nn.CrossEntropyLoss()

optimiser_ft = optim.Adam(model.parameters(), lr=0.001)

exp_lr_scheduler = lr_scheduler.StepLR(optimiser_ft, step_size=7, gamma=0.1)
def train_model(model, criterion, optimiser, scheduler, num_epochs=25):

    

    model = model.to(device)

    

    best_acc = 0.0

    best_model_wts = copy.deepcopy(model.state_dict())

    

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-'*10)

        

        for phase in ['train', 'test']:

            if phase == 'train':

                scheduler.step()

                model.train()

            

            else:

                model.eval()

            

            running_loss = 0.0

            running_corrects = 0

            

            for inputs, labels in dataloaders[phase]:

                

                

            

                inputs = inputs.to(device)

                labels = labels.to(device)

            

                optimiser.zero_grad()

                

            

                with torch.set_grad_enabled(phase=='train'):

                

                    outputs = model(inputs)

                

                    _,preds = torch.max(outputs, 1)

                

                    loss = criterion(outputs, labels)

                    

                

                    if phase == 'train':

                        loss.backward()

                        optimiser.step()

                        

                    

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds==labels.data)

                

                

            

        epoch_loss = running_loss/total_batch_sizes[phase]

        epoch_acc = running_corrects.double()/(total_batch_sizes[phase] * batch_size)

        

        print(phase, epoch_loss, epoch_acc)

        

        if phase == 'test' and epoch_acc > best_acc:

            best_acc = epoch_acc

            best_model_wts = copy.deepcopy(model.state_dict())

            

    print('Training complete')

    

    print('Best val Acc: {:4f}'.format(best_acc))

    

    model.load_state_dict(best_model_wts)

    

    return model
model = train_model(model, criterion, optimiser_ft,

                   exp_lr_scheduler, num_epochs=5)
model.eval()
with torch.no_grad():

    

    correct = 0

    total = 0

    

    for images, labels in dataloaders['test']:

        

        images = images.to(device)

        labels = labels.to(device)

        

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        

    print('Accuracy of the model on the test images:', 100 * correct/total)
with torch.no_grad():

    

    inputs, labels = iter(dataloaders['test']).next()

    

    inputs = inputs.to(device)

    labels = labels.to(device)

    

    inp = torchvision.utils.make_grid(inputs)

    

    outputs = model(inputs)

    _, preds = torch.max(outputs, 1)

    

    for j in range(len(inputs)):

        inp = inputs.data[j]

        imshow(inp, 'predicted:' + class_names[preds[j]])