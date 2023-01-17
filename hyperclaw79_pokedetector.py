%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from torch.autograd import Variable
from torchvision import transforms, models
from PIL import Image
data_dir = '../input/poketwo/poketwo_dataset'
def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(datadir,       
                    transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=64)
    return trainloader, testloader
trainloader, testloader = load_split_train_test(data_dir, .2)
# model = models.resnet50(pretrained=True)
model = torch.load('../input/pokedetector/pokemodel2.pth')
# for param in model.parameters():
#     param.requires_grad = False
# Freeze all layers except last two.
for child in list(model.children())[:-2]:
    for param in child.parameters():
        param.requires_grad = False
# model.fc = nn.Sequential(
#     nn.Linear(model.fc.in_features, 512),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(512, len(os.listdir('../input/poketwo/poketwo_dataset'))),
#     nn.LogSoftmax(dim=1)
# )
# optimizer = optim.Adam(model.fc.parameters(), lr=0.0003)

optimizer = optim.Adam(model.parameters(), lr=0.0003)
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "cpu"
)
criterion = nn.NLLLoss()
model.to(device)
epochs = 60
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
    torch.save(model, 'pokemodel3.pth')
    if accuracy / len(testloader) >= 0.999:
        print('Achieved maximum accuracy!')
        break
import os
import random
import torch

from torch.autograd import Variable
from torchvision import transforms, models
from PIL import Image


class PokeDetector:
    def __init__(self, classes, model_path='pokemodel3.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path)
        self.model.eval()
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.classes = classes

    def predict(self, image_path):
        image = Image.open(image_path)
        image = self.transforms(image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        output = self.model(image)
        index = output.data.cpu().numpy().argmax()
        return str(self.classes[index])

classes = sorted(os.listdir('../input/poketwo/poketwo_dataset'))
detector = PokeDetector(classes=classes, model_path='../input/pokedetector/pokemodel3.pth')

for i in range(100):
    test = random.choice(os.listdir('../input/poketwo/poketwo_dataset'))
    print(f"[{i+1}] Testing for {test}.")
    prediction = detector.predict(f'../input/poketwo/poketwo_dataset/{test}/{random.randint(0, 10)}.png')
    print(f"Result: {prediction}")

