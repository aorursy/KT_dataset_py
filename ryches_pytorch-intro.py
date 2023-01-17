import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
x = torch.empty(5,3)

print(x)
x = torch.rand(5, 3)

print(x)
x = torch.zeros(5, 3, dtype=torch.long)

print(x)
x = torch.tensor([5.5, 3])

print(x)
print(x.size())
x = torch.tensor([[5.5, 3], [5.5, 3]])

print(x)
x.size()
np.zeros((2,2)).shape
x = torch.rand(5, 3)

y = torch.rand(5, 3)

print(x + y)
print(torch.add(x, y))
result = torch.empty(5, 3)

torch.add(x, y, out=result)

print(result)
x = torch.randn(4, 4)

y = x.view(16)

z = x.view(-1, 8)  # the size -1 is inferred from other dimensions

print(x.size(), y.size(), z.size())
np_zeros = np.zeros((4,4))
np_zeros.reshape((-1,8))
z[0, 0].item()
x.numpy() + np_zeros
x = torch.zeros((4,4), dtype = torch.double)
x + torch.from_numpy(np_zeros)
torch.cuda.is_available()
y = x.cuda()
%%time

y + y
%%time

x + x
y = torch.randn(500,200)

w = torch.randn(200,500)
%%time

torch.mm(y,w)
y = y.cuda()

w = w.cuda()
%%time

torch.mm(y,w)
x = torch.ones(2, 2, requires_grad=True)

print(x)
y = x + 2

print(y)
z = y * y * 3

out = z.mean()



print(z, out)
a = torch.randn(2, 2)

a = ((a * 3) / (a - 1))

print(a.requires_grad)

a.requires_grad_(True)

print(a.requires_grad)

b = (a * a).sum()

print(b.grad_fn)
out.backward()
print(x.grad)
print(x.requires_grad)

print((x ** 2).requires_grad)



with torch.no_grad():

    print((x ** 2).requires_grad)
import torch

import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):



    def __init__(self):

        super(Net, self).__init__()

        # 1 input image channel, 6 output channels, 3x3 square convolution

        # kernel

        self.conv1 = nn.Conv2d(1, 6, 3)

        self.conv2 = nn.Conv2d(6, 16, 3)

        # an affine operation: y = Wx + b

        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

        # Max pooling over a (2, 2) window

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # If the size is a square you can only specify a single number

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x



    def num_flat_features(self, x):

        size = x.size()[1:]  # all dimensions except the batch dimension

        num_features = 1

        for s in size:

            num_features *= s

        return num_features





net = Net()

print(net)
params = list(net.parameters())

print(len(params))

print(params[0].size())  # conv1's .weight
input = torch.randn(1, 1, 32, 32)

out = net(input)

print(out)
net.zero_grad()

out.backward(torch.randn(1, 10))
output = net(input)

target = torch.randn(10)  # a dummy target, for example

target = target.view(1, -1)  # make it the same shape as output

criterion = nn.MSELoss()



loss = criterion(output, target)

print(loss)
net.zero_grad()     # zeroes the gradient buffers of all parameters



print('conv1.bias.grad before backward')

print(net.conv1.bias.grad)



loss.backward()



print('conv1.bias.grad after backward')

print(net.conv1.bias.grad)
learning_rate = 0.001

for f in net.parameters():

    f.data.sub_(f.grad.data * learning_rate)
import torch.optim as optim



# create your optimizer

optimizer = optim.Adam(net.parameters(), lr=0.01)



# in your training loop:

optimizer.zero_grad()   # zero the gradient buffers

output = net(input)

loss = criterion(output, target)

loss.backward()

optimizer.step()    # Does the update
import torch

import torchvision

import torchvision.transforms as transforms
transform = transforms.Compose(

    [transforms.ToTensor(),

     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



trainset = torchvision.datasets.CIFAR10(root='./data', train=True,

                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,

                                          shuffle=True, num_workers=2)



testset = torchvision.datasets.CIFAR10(root='./data', train=False,

                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,

                                         shuffle=False, num_workers=2)



classes = ('plane', 'car', 'bird', 'cat',

           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
import matplotlib.pyplot as plt

import numpy as np



# functions to show an image





def imshow(img):

    img = img / 2 + 0.5     # unnormalize

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()





# get some random training images

dataiter = iter(trainloader)

images, labels = dataiter.next()



# show images

imshow(torchvision.utils.make_grid(images))

# print labels

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x





net = Net()
import torch.optim as optim



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
%%time

for epoch in range(2):  # loop over the dataset multiple times



    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data



        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        # print statistics

        running_loss += loss.item()

        if i % 2000 == 1999:    # print every 2000 mini-batches

            print('[%d, %5d] loss: %.3f' %

                  (epoch + 1, i + 1, running_loss / 2000))

            running_loss = 0.0



print('Finished Training')
dataiter = iter(testloader)

images, labels = dataiter.next()



# print images

imshow(torchvision.utils.make_grid(images))

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
outputs = net(images)
_, predicted = torch.max(outputs, 1)



print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]

                              for j in range(4)))
correct = 0

total = 0

with torch.no_grad():

    for data in testloader:

        images, labels = data

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network on the 10000 test images: %d %%' % (

    100 * correct / total))
images.shape
class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))

with torch.no_grad():

    for data in testloader:

        images, labels = data

        outputs = net(images)

        _, predicted = torch.max(outputs, 1)

        c = (predicted == labels).squeeze()

        for i in range(4):

            label = labels[i]

            class_correct[label] += c[i].item()

            class_total[label] += 1





for i in range(10):

    print('Accuracy of %5s : %2d %%' % (

        classes[i], 100 * class_correct[i] / class_total[i]))
#uncomment to see what the state_dict looks like. All of the weights of the network

# net.state_dict()
torch.save(net.state_dict(), "model.pth")
model = Net()

model.load_state_dict(torch.load("model.pth"))

model.eval()
torch.save({

            'epoch': epoch,

            'model_state_dict': model.state_dict(),

            'optimizer_state_dict': optimizer.state_dict(),

            'loss': loss,

            }, "full_model.pth")

checkpoint = torch.load("full_model.pth")

model.load_state_dict(checkpoint['model_state_dict'])

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

epoch = checkpoint['epoch']

loss = checkpoint['loss']



model.eval()

# - or -

model.train()
torch.save(model, "standalone_model.pth")

model = torch.load("standalone_model.pth")
class SimpleNet(nn.Module):



    def __init__(self):

        super(SimpleNet, self).__init__()

        self.fc1 = nn.Linear(32*32, 100)



        # an affine operation: y = Wx + b

        self.fc2 = nn.Linear(100, 60)

        self.fc3 = nn.Linear(60, 32)

        self.fc4 = nn.Linear(32, 10)



    def forward(self, x):

        x = self.fc1(x)

        x = F.relu(x)

        x = self.fc2(x)

        x = F.relu(x)

        x = self.fc3(x)

        x = F.relu(x)

        x = self.fc4(x)

        x = F.relu(x)

        return x





net = SimpleNet()

print(net)
# License: BSD

# Author: Sasank Chilamkurthy



from __future__ import print_function, division



import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

import numpy as np

import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import time

import os

import copy



plt.ion()   # interactive mode
os.listdir("../input/hymenoptera-data/hymenoptera_data/hymenoptera_data/")
# Data augmentation and normalization for training

# Just normalization for validation

data_transforms = {

    'train': transforms.Compose([

        transforms.RandomResizedCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'val': transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

}



data_dir = '../input/hymenoptera-data/hymenoptera_data/hymenoptera_data/'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),

                                          data_transforms[x])

                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,

                                             shuffle=True, num_workers=4)

              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def imshow(inp, title=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated





# Get a batch of training data

inputs, classes = next(iter(dataloaders['train']))



# Make a grid from batch

out = torchvision.utils.make_grid(inputs)



imshow(out, title=[class_names[x] for x in classes])
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            if phase == 'train':

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0



            # Iterate over data.

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)

                labels = labels.to(device)



                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                # statistics

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':

                scheduler.step()



            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects.double() / dataset_sizes[phase]



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                phase, epoch_loss, epoch_acc))



            # deep copy the model

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model
def visualize_model(model, num_images=6):

    was_training = model.training

    model.eval()

    images_so_far = 0

    fig = plt.figure()



    with torch.no_grad():

        for i, (inputs, labels) in enumerate(dataloaders['val']):

            inputs = inputs.to(device)

            labels = labels.to(device)



            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)



            for j in range(inputs.size()[0]):

                images_so_far += 1

                ax = plt.subplot(num_images//2, 2, images_so_far)

                ax.axis('off')

                ax.set_title('predicted: {}'.format(class_names[preds[j]]))

                imshow(inputs.cpu().data[j])



                if images_so_far == num_images:

                    model.train(mode=was_training)

                    return

        model.train(mode=was_training)
model_ft = models.resnet18(pretrained=True)

num_ftrs = model_ft.fc.in_features

# Here the size of each output sample is set to 2.

# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).

model_ft.fc = nn.Linear(num_ftrs, 2)



model_ft = model_ft.to(device)



criterion = nn.CrossEntropyLoss()



# Observe that all parameters are being optimized

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,

                       num_epochs=25)
visualize_model(model_ft)
model_conv = torchvision.models.resnet18(pretrained=True)

for param in model_conv.parameters():

    param.requires_grad = False



# Parameters of newly constructed modules have requires_grad=True by default

num_ftrs = model_conv.fc.in_features

model_conv.fc = nn.Linear(num_ftrs, 2)



model_conv = model_conv.to(device)



criterion = nn.CrossEntropyLoss()



# Observe that only parameters of final layer are being optimized as

# opposed to before.

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
model_conv = train_model(model_conv, criterion, optimizer_conv,

                         exp_lr_scheduler, num_epochs=25)