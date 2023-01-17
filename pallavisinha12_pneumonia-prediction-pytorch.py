import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

import torch

import torch.nn as nn

import torchvision

import torchvision.datasets as datasets

import torchvision.models as models

import torchvision.transforms as transforms

import copy

import time

import torch.optim as optim
from torchvision import transforms

transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])
trainset = torchvision.datasets.ImageFolder("../input/chest-xray-pneumonia/chest_xray/train", transform = transform)

valset = torchvision.datasets.ImageFolder("../input/chest-xray-pneumonia/chest_xray/val", transform = transform)

testset = torchvision.datasets.ImageFolder("../input/chest-xray-pneumonia/chest_xray/test", transform = transform)
batch_size = 128

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

evalloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
dataiter = iter(trainloader)

images, labels = dataiter.next()

print(images.shape)

print(labels.shape)
classes = ['Normal', 'Pneumonia']
model = torchvision.models.resnet50(pretrained=True, progress=True)
print(model)
for p in model.parameters():

    p.requires_grad = False

model.fc = nn.Sequential(

    nn.Linear(2048, 1024),

    nn.ReLU(),

    nn.Linear(1024, 512),

    nn.ReLU(),

    nn.Linear(512, 256),

    nn.ReLU(),

    nn.Linear(256,2)

    

)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()

opt = optim.Adam(model.parameters(), lr = 0.001)
def evaluation(dataloader):

    total, correct = 0,0

    model.eval()

    for data in dataloader:

        images, labels = data

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        _, pred = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (pred == labels).sum().item()

    return 100*correct/total
%%time

loss_train_arr = []

loss_val_arr = []



max_epochs = 10

min_val_loss = 1000

for epoch in range(max_epochs):

    for i, data in enumerate(trainloader, 0):

        inputs,labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        opt.zero_grad()

        model.train()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)

        loss.backward()

        opt.step()

        model.eval()

        for j in evalloader:

            images_val, labels_val = j

            images_val, labels_val = images_val.to(device), labels_val.to(device)

            output_val = model(images_val)

            val_loss = loss_fn(output_val, labels_val)

            if min_val_loss > val_loss.item():

                min_val_loss = val_loss.item()

                best_model = copy.deepcopy(model.state_dict())

                print('Train loss %0.2f' % loss.item(), end = " ")

                print('Min val loss %0.2f' % min_val_loss)  

                del inputs, labels, outputs, images_val, labels_val, output_val

                torch.cuda.empty_cache()

    loss_val_arr.append(val_loss.item())

    loss_train_arr.append(loss.item())

    model.eval()

        

    print('Epoch: %d/%d, Train acc: %0.2f, eval acc: %0.2f' % (

        epoch+1, max_epochs, 

        evaluation(trainloader), evaluation(evalloader)))

    

%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt



plt.plot(loss_train_arr, label='Training Loss')

plt.plot(loss_val_arr, label='Validation Loss')

plt.legend(frameon=False)
model.load_state_dict(best_model)

print(evaluation(testloader))
classes = ['Normal', 'Pneumonia']

dataiter = iter(testloader)

images, labels = dataiter.next()

def imshow(img):

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1,2,0)))

    plt.show()

imshow(torchvision.utils.make_grid(images[:1]))

print("Ground_Truth-")

print(classes[labels[0]])

images = images.to(device)

output = model(images)

max_values, pred_class = torch.max(output.data, 1)

print("Predicted class-")

print(classes[pred_class[0]])