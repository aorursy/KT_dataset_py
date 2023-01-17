import torch

import torchvision

from torchvision import transforms, datasets

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim
# Training data

train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))



# Testing data

test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))



# Loading the datasets

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)
# Creating a class with a forward NN with 3 hidden layers with 64 output size



class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(28*28, 64)

        self.fc2 = nn.Linear(64, 64)

        self.fc3 = nn.Linear(64, 64)

        self.fc4 = nn.Linear(64, 10)

    

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = F.log_softmax(self.fc4(x), dim=1)

        

        return x



net = Net()

print(net)
# Using Adam optimizer to optimize weights of the Neural Network



optimizer = optim.Adam(net.parameters(), lr =0.001)



EPOCHS = 3



for epoch in range(EPOCHS):

    for data in trainset:

        X, y = data

        net.zero_grad()

        output = net(X.view(-1, 28*28))

        loss = F.nll_loss(output, y)

        loss.backward()

        optimizer.step()

    print(loss)
# Calculating Accuracy



correct = 0

total = 0



with torch.no_grad():

     for data in trainset:

            X, y = data

            output = net(X.view(-1, 784))

            for idx, i in enumerate(output):

                if torch.argmax(i) == y[idx]:

                    correct += 1

                total +=1

                

print("Accuracy: ", round(correct/total, 3))