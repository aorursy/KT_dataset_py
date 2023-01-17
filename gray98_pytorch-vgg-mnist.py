import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torchvision import datasets, transforms



print("PyTorch Version: ",torch.__version__)
class Net(nn.Module):

    def __init__(self):

        super(Net,self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)

        

        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)

        

        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)

        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)

        

        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)

        self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)

        

        self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)

        self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)

        

        self.fc1 = nn.Linear(7*7*512, 4096)

        self.fc2 = nn.Linear(4096, 4096)

        self.fc3 = nn.Linear(4096, 10)

        self.drop = nn.Dropout(0.5)

    def forward(self,x):

        x = F.relu(self.conv1(x)) #224

        x = F.max_pool2d(x, 2, 2) #112

        

        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2) #56

        

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = F.max_pool2d(x, 2, 2) # 28

        

        x = F.relu(self.conv5(x))

        x = F.relu(self.conv6(x))

        x = F.max_pool2d(x, 2, 2) # 14

        

        x = F.relu(self.conv7(x))

        x = F.relu(self.conv8(x))

        x = F.max_pool2d(x, 2, 2) #7

        

        x = x.view(-1, 7*7*512)

        x = self.drop(F.relu(self.fc1(x)))

        x = self.drop(F.relu(self.fc2(x)))

        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

# class VGG16(nn.Module):

#     def __init__(self):

#         super(Net, self).__init__()

#         self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)

#         self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)

        

#         self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)

#         self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)

        

#         self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)

#         self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)

#         self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)

        

#         self.conv8 = nn.Conv2d(256, 512, 3, 1, 1)

#         self.conv9 = nn.Conv2d(512, 512, 3, 1, 1)

#         self.conv10 = nn.Conv2d(512, 512, 3, 1, 1)

        

#         self.conv11 = nn.Conv2d(512, 512, 3, 1, 1)

#         self.conv12 = nn.Conv2d(512, 512, 3, 1, 1)

#         self.conv13 = nn.Conv2d(512, 512, 3, 1, 1)

        

#         self.fc1 = nn.Linear(7*7*512, 4096)

#         self.fc2 = nn.Linear(4096, 4096)

#         self.fc3 = nn.Linear(4096, 10)

        

#     def forward(self, x):

#         x = F.relu(self.conv1(x)) 

#         x = F.relu(self.conv2(x)) 

#         x = F.max_pool2d(x,2,2) 

        

#         x = F.relu(self.conv3(x)) 

#         x = F.relu(self.conv4(x)) 

#         x = F.max_pool2d(x,2,2) 

        

#         x = F.relu(self.conv5(x))

#         x = F.relu(self.conv6(x))

#         x = F.relu(self.conv7(x))

#         x = F.max_pool2d(x,2,2)

        

#         x = F.relu(self.conv8(x))

#         x = F.relu(self.conv9(x))

#         x = F.relu(self.conv10(x))

#         x = F.max_pool2d(x,2,2)

        

#         x = F.relu(self.conv11(x))

#         x = F.relu(self.conv12(x))

#         x = F.relu(self.conv13(x))

#         x = F.max_pool2d(x,2,2)

        

#         x = x.view(-1, 7*7*512) 

#         x = F.relu(self.fc1(x))

#         x = F.relu(self.fc2(x))

#         x = self.fc3(x)

#         return F.log_softmax(x, dim=1)
transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

batch_size = 32

train_dataloader = torch.utils.data.DataLoader(

    datasets.MNIST("./mnist_data", train=True, download=True,

                   transform=transform),batch_size=batch_size, shuffle=True, num_workers=0)

test_dataloader = torch.utils.data.DataLoader(

    datasets.MNIST("./mnist_data",train=False, download=True,

                  transform=transform), batch_size=batch_size, shuffle=True, num_workers=0)

mnist_data = datasets.MNIST("./mnist_data", train=True, download=True,transform=transforms.Compose([transforms.Resize(224)]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device
def train(model, device, train_loader, optimizer, epoch):

    model.train()

    for idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        

        pred = model(data)

        loss = F.nll_loss(pred, target) # cross_entropy

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        if idx % 100 == 0:

            print("Train Epoch: {}, iteration: {}, loss: {}".format(epoch, idx, loss.item()))

def test(model, device, test_loader):

    model.eval()

    total_loss = 0.

    correct = 0.

    with torch.no_grad():

        for idx, (data, target) in enumerate(test_loader):

            data, target = data.to(device), target.to(device)



            output = model(data)

            total_loss += F.nll_loss(output, target, reduction="sum").item() # cross_entropy

            pred = output.argmax(dim=1)

            correct += pred.eq(target.view_as(pred)).sum().item()

    

    total_loss /= len(test_loader.dataset)

    acc = correct/len(test_loader.dataset) * 100.

    print("Test loss: {}, Accuracy: {}".format(total_loss, acc))
lr = 0.01

momentum = 0.5

model = Net().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

num_epoches = 5

for epoch in range(num_epoches):

    train(model, device, train_dataloader, optimizer, epoch)

    test(model, device, test_dataloader)

torch.save(model.state_dict(), "mnist_cnn.pt")