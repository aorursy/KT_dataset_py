import torch.nn as nn

import numpy as np

import torch as tc





class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        np.random.seed(1) 

        tc.manual_seed(1)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)

        self.layer1 = nn.Sequential(self.conv1, nn.ReLU(),

                                    nn.MaxPool2d(kernel_size=2, stride=2))

        

        self.conv2=nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)

        self.layer2 = nn.Sequential(self.conv2, nn.ReLU(),

                                    nn.MaxPool2d(kernel_size=2, stride=2))

        

        self.drop_out = nn.Dropout()

        self.fc1 = nn.Linear(25 * 25 * 128, 300)

        self.fc2 = nn.Linear(300, 103)



    def forward(self, x):

        out = self.layer1(x)

        out = self.layer2(out)

        out = out.reshape(out.size(0), -1)

        out = self.drop_out(out)

        out = self.fc1(out)

        out = self.fc2(out)

        return out

import torch as tc

import torch.nn as nn

import torch.optim as optim

import torchvision

from torchvision import transforms

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

import numpy as np



np.random.seed(1) 

tc.manual_seed(1)



def train_preparation():

    train_data_dir = '../input/fruits-360_dataset/fruits-360/Training/'

    TRANSFORM_IMG = transforms.Compose(

        [transforms.ToTensor(),

         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



    train_folder = torchvision.datasets.ImageFolder(

            root=train_data_dir,

            transform = TRANSFORM_IMG

        )

    train_loader = tc.utils.data.DataLoader(

            train_folder,

            batch_size=16,

            shuffle=True

        )

    return train_loader



def test_preparation():

    test_data_dir = '../input/fruits-360_dataset/fruits-360/Test/'

    TRANSFORM_IMG = transforms.Compose(

        [transforms.ToTensor(),

         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



    test_folder = torchvision.datasets.ImageFolder(

        root=test_data_dir,

        transform=TRANSFORM_IMG

    )

    test_set, valid_set = train_test_split(test_folder,test_size=0.2)

    test_loader = tc.utils.data.DataLoader(

        test_set,

        batch_size=16,

        shuffle=True

    )

    valid_loader = tc.utils.data.DataLoader(

        valid_set,

        batch_size=16,

        shuffle=True

    )

    return test_loader,valid_loader



def training(train_loader,valid_loader):

    device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")

    net = Net()

    net.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)



    for epoch in range(4):



        total_step = len(train_loader)

        loss_list = []

        acc_list = []

        for i, data in enumerate(train_loader, 0):



            inputs, labels = data

            inputs = inputs.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()



            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss_list.append(loss.item())

            loss.backward()

            optimizer.step()



            total = labels.size(0)

            _, predicted = tc.max(outputs.data, 1)

            correct = (predicted == labels).sum().item()

            acc_list.append(correct / total)



            if (i + 1) % 100 == 0:

                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'

                      .format(epoch + 1, 4, i + 1, total_step, loss.item(),

                              (correct / total) * 100))

                

                if  validating(net,valid_loader) >= 89:

                    print('Finished Training')

                    return net

    print('Finished Training')

    return net





def validating(net,test_loader):

    device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")

    net.eval()

    with tc.no_grad():

        correct = 0

        total = 0

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = net(images)

            _, predicted = tc.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            



        print('Test Accuracy of the model: {} %'.format((correct / total) * 100))

        return (correct / total) * 100



def testing(net,test_loader):

    device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")

    net.eval()

    valid_preds = []

    valid_truth = []

    with tc.no_grad():

        correct = 0

        total = 0

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = net(images)

            _, predicted = tc.max(outputs.data, 1)

            valid_preds += predicted.tolist()

            valid_truth += labels.tolist()

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            



        print('Test Accuracy of the model: {} %'.format((correct / total) * 100))

        return (correct / total) * 100,valid_preds,valid_truth

        

train_loader= train_preparation()

test_loader,valid_loader  = test_preparation()

net = training(train_loader,valid_loader)

score,preds,labels = testing(net,test_loader)

print(classification_report(labels,preds))