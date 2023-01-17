import torch

import torchvision

import torchvision.transforms as transforms



transform = transforms.Compose(

    [transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

)



# ao usar o Kaggle, o root deve apontar para '../input/cifar10_pytorch/data'



trainset = torchvision.datasets.CIFAR10(root='../input/cifar10-pytorch/cifar10_pytorch/data', train=True,

                                        download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,

                                          shuffle=True, num_workers=2)



testset = torchvision.datasets.CIFAR10(root='../input/cifar10-pytorch/cifar10_pytorch/data', train=False,

                                       download=False, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,

                                         shuffle=False, num_workers=2)



classes = ('plane', 'car', 'bird', 'cat',

           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
import torch.nn as nn

import torch.nn.functional as F



#adicionado uma nova camada de convolução após a normalização



class Net1(nn.Module):

    def __init__(self):

        super(Net1, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.conv2_bn = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 26, 5)

        self.fc1 = nn.Linear(26 * 1 * 1, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = self.conv2_bn(x)

        x = F.relu(self.conv3(x))

        x = x.view(-1, 26 * 1 * 1)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x





net1 = Net1()
#Removido os pools



class Net2(nn.Module):

    def __init__(self):

        super(Net2, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.conv2_bn = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 26, 5)

        self.fc1 = nn.Linear(26 * 20 * 20, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = self.conv2_bn(x)   

        x = F.relu(self.conv3(x))

        x = x.view(-1, 26 * 20 * 20)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x





net2 = Net2()
#aplicado pool apenas quando concluido todas as 3 convoluções



class Net3(nn.Module):

    def __init__(self):

        super(Net3, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.conv2_bn = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 26, 5)

        self.fc1 = nn.Linear(26 * 10 * 10, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = self.conv2_bn(x)

        x = F.relu(self.conv3(x))

        x = self.pool(x)

        x = x.view(-1, 26 * 10 * 10)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x





net3 = Net3()
import torch.optim as optim



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



nets = [net1, net2, net3]

optimizers = [

    optim.SGD(net1.parameters(), lr=0.001, momentum=0.9), 

    optim.SGD(net2.parameters(), lr=0.001, momentum=0.9), 

    optim.SGD(net3.parameters(), lr=0.001, momentum=0.9)

 ]



for j in range(3):

    print("Net #%d" % (j+1))

    net = nets[j]

    net.to(device)

    media = 0

    

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optimizers[j]

    

    for k in range(5):

        

        for epoch in range(2):  # loop over the dataset multiple times



            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):

                # get the inputs; data is a list of [inputs, labels]

                inputs, labels = data[0].to(device), data[1].to(device)



                # zero the parameter gradients

                optimizer.zero_grad()



                # forward + backward + optimize

                outputs = net(inputs)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()



        ###Teste



        correct = 0

        total = 0

        with torch.no_grad():

            for data in testloader:

                images, labels = data[0].to(device), data[1].to(device)

                outputs = net(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()



        acuracia = (100 * correct / total)

        media = media + acuracia

        print('Accuracy of the network on the 10000 test images: %d %%' % acuracia)

    print("Average Acurracy: %d" % (media/5))