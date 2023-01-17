import matplotlib.pyplot as plt

import numpy as np

import time



import torch

import torchvision

import torchvision.transforms as transforms



import torch.nn as nn

import torch.nn.functional as F



import torch.optim as optim
transform = transforms.Compose(

    [transforms.ToTensor(),

     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



batch_size=32

epochs=10



trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)



testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)



classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
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

print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3)



        self.fc1 = nn.Linear(32 * 6 * 6, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        #print(x.shape)

        x = self.pool(F.relu(self.conv2(x)))

        #print(x.shape)

        x = x.view(-1, 32 * 6 * 6)

        #print(x.shape)

        x = F.relu(self.fc1(x))

        #print(x.shape)

        x = F.relu(self.fc2(x))

        #print(x.shape)

        x = self.fc3(x)

        #print(x.shape)

        #quebra

        return x
net = Net()

net
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

optimizer = optim.AdamW(net.parameters(), lr=0.01)
def train(net, criterion, optimizer):

    since = time.time()



    for epoch in range(epochs):  # loop over the dataset multiple times



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

            if i % 100 == 99:    # print every 2000 mini-batches

                print('[%d, %5d] loss: %.3f - time: %.2f' % (epoch + 1, i + 1, running_loss / 2000, time.time() - since))

                running_loss = 0.0



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))



train(net, criterion, optimizer)
dataiter = iter(testloader)

images, labels = dataiter.next()



# print images

imshow(torchvision.utils.make_grid(images))

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
outputs = net(images)

_, predicted = torch.max(outputs, 1)



print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))
correct = 0

total = 0

with torch.no_grad():

    for data in testloader:

        images, labels = data

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
def eval_net(net):

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

        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

        

eval_net(net)
net = Net()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
def train_gpu(net, criterion, optimizer):

    since = time.time()



    ###

    net.to(device)



    for epoch in range(epochs):  # loop over the dataset multiple times



        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]

            ### inputs, labels = data

            inputs, labels = data[0].to(device), data[1].to(device)





            # zero the parameter gradients

            optimizer.zero_grad()



            # forward + backward + optimize

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()



            # print statistics

            running_loss += loss.item()

            if i % 100 == 99:    # print every 2000 mini-batches

                print('[%d, %5d] loss: %.3f - time: %.2f' % (epoch + 1, i + 1, running_loss / 2000, time.time() - since))

                running_loss = 0.0



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))



train_gpu(net, criterion, optimizer)
correct = 0

total = 0

with torch.no_grad():

    for data in testloader:

        images, labels = data[0].to(device), data[1].to(device)



        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
def eval_net_gpu(net):

    class_correct = list(0. for i in range(10))

    class_total = list(0. for i in range(10))

    with torch.no_grad():

        for data in testloader:

            images, labels = data[0].to(device), data[1].to(device)



            outputs = net(images)

            _, predicted = torch.max(outputs, 1)

            c = (predicted == labels).squeeze()

            for i in range(4):

                label = labels[i]

                class_correct[label] += c[i].item()

                class_total[label] += 1



    for i in range(10):

        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

        

eval_net_gpu(net)
# referencias

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py