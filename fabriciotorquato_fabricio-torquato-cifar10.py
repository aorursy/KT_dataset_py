import torch

import torchvision

import torchvision.transforms as transforms

import torch.optim as optim

import torch.nn as nn

import torch.nn.functional as F



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




class NetOrginal(nn.Module):

    def __init__(self):

        super(NetOrginal, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.conv2_bn = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool(x)

        x = self.conv2_bn(x)

        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x
class CatAndDogNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)

        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(3, 3), padding=1)

        self.conv3_bn = nn.BatchNorm2d(32)



        self.fc1 = nn.Linear(in_features= 32 * 3 * 3, out_features=500)

        self.dropout = nn.Dropout(0.5)        

        self.fc2 = nn.Linear(in_features=500, out_features=50)

        self.fc3 = nn.Linear(in_features=50, out_features=10)



        

    def forward(self, X):

        X = F.relu(self.conv1(X))

        X = F.max_pool2d(X, 2)

        

        X = F.relu(self.conv3_bn(self.conv3(X)))

        X = F.max_pool2d(X, 2)

        

#         print(X.shape)

        X = X.view(X.shape[0], -1)

        X = F.relu(self.fc1(X))

        X = self.dropout(X)

        X = F.relu(self.fc2(X))

        X = self.fc3(X)

        

#         X = torch.sigmoid(X)

        return X
class CnnTCC(nn.Module):

    def __init__(self):

        super(CnnTCC, self).__init__()

        self.output_layer = 10

        self.conv0 = nn.Sequential(

            nn.Conv2d(3, 8, 8, 1, 1),

            nn.BatchNorm2d(8, False)

        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(8, 16, 8, 1, 1),

            nn.BatchNorm2d(16, False),

            nn.MaxPool2d(2)

        )

        self.dense = nn.Sequential(

            nn.Linear(4 * 22 * 22, 32),

            nn.Tanh(),

            nn.Linear(32, 32),

            nn.Tanh(),

            nn.Linear(32, self.output_layer),

        )



    def forward(self, x):

        x = self.conv0(x)

        x = self.conv2(x)

        x = x.view(x.size(0), -1)

        x = self.dense(x)

        return F.softmax(x, dim=1)
class CNN_MNIST(torch.nn.Module):



    def __init__(self):

        super(CNN_MNIST, self).__init__()

        # L1 ImgIn shape=(?, 28, 28, 1)

        #    Conv     -> (?, 28, 28, 32)

        #    Pool     -> (?, 14, 14, 32)

        self.layer1 = torch.nn.Sequential(

            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),

            torch.nn.ReLU(),

            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Dropout(p=0.1))

        # L2 ImgIn shape=(?, 14, 14, 32)

        #    Conv      ->(?, 14, 14, 64)

        #    Pool      ->(?, 7, 7, 64)

        self.layer2 = torch.nn.Sequential(

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),

            torch.nn.ReLU(),

            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Dropout(p=0.1))

        # L3 ImgIn shape=(?, 7, 7, 64)

        #    Conv      ->(?, 7, 7, 128)

        #    Pool      ->(?, 4, 4, 128)

        self.layer3 = torch.nn.Sequential(

            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),

            torch.nn.ReLU(),

            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            torch.nn.Dropout(p=0.1))



        # L4 FC 3200 inputs -> 625 outputs

        self.fc1 = torch.nn.Linear(3200, 625, bias=True)

        torch.nn.init.xavier_uniform(self.fc1.weight)

        self.layer4 = torch.nn.Sequential(

            self.fc1,

            torch.nn.ReLU(),

            torch.nn.Dropout(p=0.1))

        # L5 Final FC 625 inputs -> 10 outputs

        self.fc2 = torch.nn.Linear(625, 10, bias=True)

        torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters



    def forward(self, x):

        out = self.layer1(x)

        out = self.layer2(out)

        out = self.layer3(out)

        out = out.view(out.size(0), -1)   # Flatten them for FC

        out = self.fc1(out)

        out = self.fc2(out)

        return out
def train(net):



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.to(device)



    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



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



            # print statistics

            running_loss += loss.item()

            if i % 2000 == 1999:    # print every 2000 mini-batches

                print('[%d, %5d] loss: %.3f' %

                      (epoch + 1, i + 1, running_loss / 2000))

                running_loss = 0.0



    print('Finished Training')

    return net
def test(net):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    correct = 0

    total = 0



    with torch.no_grad():

        for data in testloader:

            images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()



    print('Accuracy of the network on the 10000 test images: %d %%' % (

        100 * correct / total))

    return correct / total
# display(print("Net Original"))



# times = 5

# mean = 0



# for _ in range(times):

#     net_orginal = NetOrginal()

#     net_orginal = train(net_orginal)

#     mean += test(net_orginal)



# display(print('Accuracy means: %d %%' % (

#     100 * mean / times)))

    
# display(print("Net Cat dog"))



# times = 5

# mean = 0



# for _ in range(times):

#     net_cat_dog = CatAndDogNet()

#     net_cat_dog = train(net_cat_dog)

#     mean += test(net_cat_dog)



# display(print('Accuracy means: %d %%' % (

#     100 * mean / times)))
# display(print("Net Cat dog"))



# times = 5

# mean = 0



# for _ in range(times):

#     net_cnn_tcc = CnnTCC()

#     net_cnn_tcc = train(net_cnn_tcc)

#     mean += test(net_cnn_tcc)



# display(print('Accuracy means: %d %%' % (

#     100 * mean / times)))
# display(print("Net CNN MNIST"))



# times = 5

# mean = 0



# for _ in range(times):

#     net_cnn_mnist = CNN_MNIST()

#     net_cnn_mnist = train(net_cnn_mnist)

#     mean += test(net_cnn_mnist)

    

# display(print('Accuracy means: %d %%' % (

#     100 * mean / times)))