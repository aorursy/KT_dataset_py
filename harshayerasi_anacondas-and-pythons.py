import torch

import torchvision

import numpy as np
def load_dataset(data_path='../input/anacondas_pythons/train'):

    train_dataset = torchvision.datasets.ImageFolder(

        root=data_path,

        transform=torchvision.transforms.ToTensor()

    )

    train_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=60,

        num_workers=0,

        shuffle=True

    )

    return train_loader
trainloader = load_dataset()

testloader = load_dataset('../input/anacondas_pythons/valid')
dataiter = iter(trainloader) # To iterate the data set

images,labels = dataiter.next() # Gives the images in the order

#labels = torch.zeros(1, 38).reshape(-1)

print(images.shape) # Printing the shape of the images.



print(images[1].shape) # Shape of each image

print(labels[1].item()) # Number of labels present.
print(type(labels))

print(labels.shape)

print(labels)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)





import torch.nn as nn



class LeNet(nn.Module):

    def __init__(self): 

        super(LeNet, self).__init__()

        self.cnn_model = nn.Sequential(

            nn.Conv2d(3, 6, 5),         # (N, 3, 300, 400) -> (N,  6, 296, 396)

            nn.Tanh(),

            nn.AvgPool2d(2, stride=2),  # (N, 6, 28, 28) -> (N,  6, 14, 14)

            nn.Conv2d(6, 16, 5),        # (N, 6, 14, 14) -> (N, 16, 10, 10)  

            nn.Tanh(),

            nn.AvgPool2d(2, stride=2)   # (N,16, 10, 10) -> (N, 16, 5, 5)

        )

        self.fc_model = nn.Sequential(

            nn.Linear(111744,120),         # (N, 400) -> (N, 120)

            nn.Tanh(),

            nn.Linear(120,84),          # (N, 120) -> (N, 84)

            nn.Tanh(),

            nn.Linear(84,10)            # (N, 84)  -> (N, 10)

        )

        

    def forward(self, x):

        x = self.cnn_model(x)

        x = x.view(x.size(0), -1)

        x = self.fc_model(x)

        return x

def evaluation(dataloader, model):

    total, correct = 0, 0

    for data in dataloader:

        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        _, pred = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (pred == labels).sum().item()

    return 100 * correct / total

import torch.optim as optim

net = LeNet().to(device)

loss_fn = nn.CrossEntropyLoss()

opt = optim.Adam(net.parameters())



import matplotlib.pyplot as plt

%%time

max_epochs = 10

loss_epoch_arr = []



for epoch in range(max_epochs):



    for i, data in enumerate(trainloader, 0):



        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)



        opt.zero_grad()



        outputs = net(inputs)

        loss = loss_fn(outputs, labels)

        loss.backward()

        opt.step()

        

    loss_epoch_arr.append(loss.item()) # Collecting loss after each epoch.

    #print('Epoch: %d/%d' % (epoch, max_epochs))

    print('Epoch: %d/%d, Test acc: %0.2f, Train acc: %0.2f, loss: %0.2f' % (epoch, max_epochs, evaluation(testloader,net), evaluation(trainloader,net),loss.item()))



plt.plot(loss_epoch_arr)

plt.show()
from torchvision.models import resnet18

import copy
resnet = resnet18(pretrained=True)
print(resnet)
for param in resnet.parameters():

    param.requires_grad = False
num_classes = 2

final_in_features = resnet.fc.in_features

resnet.fc = nn.Linear(final_in_features, num_classes)
print(resnet)
resnet = resnet.to(device)

loss_fn = nn.CrossEntropyLoss()

opt = optim.SGD(resnet.parameters(), lr=0.01)
loss_epoch_arr = []

max_epochs = 20

batch_size = 10

min_loss = 1000



n_iters = np.ceil(60/batch_size)



for epoch in range(max_epochs):



    for i, data in enumerate(trainloader, 0):



        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)



        opt.zero_grad()



        outputs = resnet(inputs)

        loss = loss_fn(outputs, labels)

        loss.backward()

        opt.step()

        

        if min_loss > loss.item():

            min_loss = loss.item()

            best_model = copy.deepcopy(resnet.state_dict())

            print('Min loss %0.2f' % min_loss)

        

        if i % 100 == 0:

            print('Iteration: %d/%d, Loss: %0.2f' % (i, n_iters, loss.item()))

            

        del inputs, labels, outputs

        torch.cuda.empty_cache()

        

    loss_epoch_arr.append(loss.item())

        

    print('Epoch: %d/%d, Test acc: %0.2f, Train acc: %0.2f' % (

        epoch, max_epochs, 

        evaluation(testloader, resnet), evaluation(trainloader, resnet)))

    

    

plt.plot(loss_epoch_arr)

plt.show()
resnet.load_state_dict(best_model)

print('The train accuracy is',evaluation(trainloader, resnet),'and the test accuracy is', evaluation(testloader, resnet))