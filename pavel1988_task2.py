# импортируем необходимые пакеты

import matplotlib.pyplot as plt

import numpy as np

import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models

%matplotlib inline

%config InlineBackend.figure_format = 'retina'
# загружаем данные в трансформер, переводим изображения в тензоры и нормализуем

data_dir = '../input/ntechdata/internship_data'

def transform_load_data(datadir, valid_size = .1):

    train_transforms = transforms.Compose([transforms.Resize((224,224)),

                                           transforms.RandomCrop([196, 196]),

                                           transforms.RandomHorizontalFlip(),

                                           transforms.ToTensor(),

                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),

                                          ])

    test_transforms = transforms.Compose([transforms.Resize((224,224)),

                                          transforms.ToTensor(),

                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),

                                         ])

    train_data = datasets.ImageFolder(datadir, transform=train_transforms)

    test_data = datasets.ImageFolder(datadir, transform=test_transforms)    

    num_train = len(train_data)

    indices = list(range(num_train))

    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    from torch.utils.data.sampler import SubsetRandomSampler

    train_idx, test_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)

    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(train_data,

                   sampler=train_sampler, batch_size=128)

    testloader = torch.utils.data.DataLoader(test_data,

                   sampler=test_sampler, batch_size=128)

    return trainloader, testloader

trainloader, testloader = transform_load_data(data_dir, .2)

print(trainloader.dataset.classes)
# подключаем gpu при необходимости и загружаем предобученную модель reset34

device = torch.device("cuda" if torch.cuda.is_available() 

                                  else "cpu")

model = models.resnet34(pretrained=True)
# устанавливаем полносвязный слой для модели 

for param in model.parameters():

    param.requires_grad = True

    

model.fc = nn.Sequential(nn.Linear(512, 256),

                                 nn.ReLU(),

                                 nn.Dropout(0.2),

                                 nn.Linear(256, 2),

                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

model.to(device)
# дообучаем модель на 2 х эпохах 

epochs = 2

steps = 0

running_loss = 0

print_every = 10

train_losses, test_losses = [], []

for epoch in range(epochs):

    for inputs, labels in trainloader:

        steps += 1

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

                for inputs, labels in testloader:

                    inputs, labels = inputs.to(device), labels.to(device)

                    logps = model.forward(inputs)

                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    

                    ps = torch.exp(logps)

                    top_p, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            train_losses.append(running_loss/len(trainloader))

            test_losses.append(test_loss/len(testloader))                    

            print(f"Epoch {epoch+1}/{epochs}.. "

                  f"Train loss: {running_loss/print_every:.3f}.. "

                  f"Test loss: {test_loss/len(testloader):.3f}.. "

                  f"Test accuracy: {accuracy/len(testloader):.3f}")

            running_loss = 0

            model.train()

torch.save(model, 'model.pth')