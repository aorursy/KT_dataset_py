import torch

import torch.nn as nn

import torchvision.datasets as datasets

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import seaborn as sns

from torch.utils.data import DataLoader

from torch.autograd import Variable

import warnings

warnings.filterwarnings('ignore')
train_dataset = datasets.MNIST(root='data/',

                              train=True,

                              transform=transforms.ToTensor(),

                              download=True)



test_dataset = datasets.MNIST(root='data/',

                             train=False,

                             transform=transforms.ToTensor())
train_dataset
test_dataset
input_size = 784

hidden_size = 400

out_size = 10

epochs = 10

batch_size = 100

learning_rate = 0.001
train_loader = DataLoader(dataset=train_dataset,

                         batch_size=batch_size,

                         shuffle=True)



valid_loader = DataLoader(dataset=test_dataset,

                         batch_size=batch_size,

                         shuffle=False)
train_dataset.classes
class Net(nn.Module):

    def __init__(self,input_size,hidden_size,out_size):

        super(Net,self).__init__()

        self.fc1 = nn.Linear(input_size,hidden_size)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size,hidden_size)

        self.fc3 = nn.Linear(hidden_size,out_size)

        

    def forward(self,x):

        out = self.fc1(x)

        out = self.relu(out)

        out = self.fc2(out)

        out = self.relu(out)

        out = self.fc3(out)

        return out
net = Net(input_size,hidden_size,out_size)
def get_default_device():

    """Pick GPU if available, else CPU"""

    if torch.cuda.is_available():

        return torch.device('cuda')

    else:

        return torch.device('cpu')
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
#visualize the train loader

for i , (img,labels) in enumerate(train_loader):

    print(img.size())

    images = img.view(-1,784)

    print(images.size())
correct_train = 0

total_train = 0

for epoch in range(epochs):

    for i, (images,label) in enumerate(train_loader):

        images = Variable(images.view(-1,28*28))

        labels = Variable(label)

        if torch.cuda.is_available():

            images = images.cuda()

            labels = labels.cuda()

        optimizer.zero_grad()

        output = net(images)

        

        _,predicted = torch.max(output.data,1)

        

        total_train += labels.size(0)

        

        

        correct_train += (predicted == labels).sum()

        

        loss = criterion(output,labels)

        loss.backward()

        optimizer.step()

        

        if (i+1) % 100 == 0:

            print(f'Epoch [{epoch}/{epochs}], Iteration [{i+1}/{len(train_dataset)//batch_size}],Training Accuracy : {100*correct_train/total_train}')

            

print('Done Training!!')
correct = 0

total = 0

for images,labels in valid_loader:

    images = Variable(images.view(-1,28*28))

    outputs = net(images)

    

    _,predict = torch.max(outputs.data,1)

    total += labels.size(0)

    

    correct += (predict == labels).sum()



print(f'Final Test Accuracy : {100*correct/total}%')