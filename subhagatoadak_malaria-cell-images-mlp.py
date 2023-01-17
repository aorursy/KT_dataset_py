# Imporing Necessary Libraries

import torch 

import numpy as np

from torch import nn, optim

import torch.nn.functional as F

from torchvision import datasets, transforms

import helper

import matplotlib.pyplot as plt

import os
# Loading the datasets from the folder, and converting the data to greyscale and resizing the data



transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize(255),

                                transforms.CenterCrop(224),

                                transforms.ToTensor()])

train_dataset = datasets.ImageFolder('../input/cell_images/cell_images/train', transform=transform)

test_dataset = datasets.ImageFolder('../input/cell_images/cell_images/test', transform=transform)

# Train loader to make iterator for the images

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
trainloader
images, labels = next(iter(trainloader))

images.shape

images.view(32,-1).shape
# View the images

plt.imshow(images[0].numpy().squeeze(), cmap='Greys_r');
# Defining the model - Multilayer Perceptron with 3 hidden layers, two classes



class classifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(50176,1000)

        self.fc2 = nn.Linear(1000,500)

        self.fc3 = nn.Linear(500,100)

        self.fc4 = nn.Linear(100,2)

        self.dropout = nn.Dropout(p=0.02)

     

    def forward(self, x):

        # make sure input tensor is flattened

        x = x.view(x.shape[0], -1)



        # Now with dropout

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))

        x = self.dropout(F.relu(self.fc3(x)))



        # output so no dropout here

        x = F.log_softmax(self.fc4(x), dim=1)



        return x
# Running the model on GPU, finding out the Training loss, Test Loss and the Accuracy



torch.manual_seed(100)

model = classifier()

criterion = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001)

model.to('cuda')

epochs = 10

steps = 0



train_losses, test_losses = [], []

for e in range(epochs):

    running_loss = 0

    for images, labels in trainloader:

        images, labels = images.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()

        

        log_ps = model(images)

        loss = criterion(log_ps, labels)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        

    else:

        test_loss = 0

        accuracy = 0

        

        # Turn off gradients for validation, saves memory and computations

        with torch.no_grad():

            model.eval()

            for images, labels in testloader:

                images, labels = images.to('cuda'), labels.to('cuda')

                log_ps = model(images)

                test_loss += criterion(log_ps, labels)

                

                ps = torch.exp(log_ps)

                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

        

        model.train()

        

        train_losses.append(running_loss/len(trainloader))

        test_losses.append(test_loss/len(testloader))



        print("Epoch: {}/{}.. ".format(e+1, epochs),

              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),

              "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),

              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))