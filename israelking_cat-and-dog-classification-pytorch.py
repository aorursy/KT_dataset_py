# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch

import matplotlib.pyplot as plt

from torchvision import datasets, transforms,models

from torch import nn,optim

import helper
directory_train = '../input/cat-and-dog/training_set/training_set'

directory_test = '../input/cat-and-dog/test_set/test_set'
train_transforms = transforms.Compose([

        transforms.RandomHorizontalFlip(),

        transforms.RandomVerticalFlip(),

        transforms.Resize(500),

        transforms.CenterCrop(500),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

train_data = datasets.ImageFolder(directory_train,transform=train_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

test_transforms = transforms.Compose([

        transforms.RandomHorizontalFlip(),

        transforms.RandomVerticalFlip(),

        transforms.Resize(500),

        transforms.CenterCrop(500),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

test_data = datasets.ImageFolder(directory_test,transform=test_transforms)

testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
class_names = trainloader.dataset.classes
class_names
image, label = next(iter(trainloader))
print(len(image))

print(len(label))
image[0].shape
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
model = models.resnet50(pretrained=True)

model.cuda()

print(model)
for param in model.parameters():

    param.requires_grad = False # Freeze pretrained model parameters to avoid backpropogating through them

    

model.fc = nn.Sequential(nn.Linear(2048, 512),

                                 nn.ReLU(),

                                 nn.Dropout(0.2),

                                 nn.Linear(512, 2),

                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

model.to(device)
epoch_list, train_accuracy_list, test_accuracy_list, train_loss_list, validation_loss_list = [], [], [], [], []

epochs = 100

for epoch in range(epochs):

    train_loss = 0

    val_loss = 0

    accuracy = 0

    # Training the model

    model.train()

    counter = 0

    for inputs, labels in trainloader:

        # Move to device

        inputs, labels = inputs.to(device), labels.to(device)

        # Clear optimizers

        optimizer.zero_grad()

        # Forward pass

        output = model.forward(inputs)

        # Loss

        loss = criterion(output, labels)

        # Calculate gradients (backpropogation)

        loss.backward()

        # Adjust parameters based on gradients

        optimizer.step()

        # Add the loss to the training set's rnning loss

        train_loss += loss.item()*inputs.size(0)

        # Print the progress

        counter += 1

    # Evaluating the model

    model.eval()

    counter = 0

    # Tell torch not to calculate gradients

    with torch.no_grad():

        for inputs, labels in testloader:

            # Move to device

            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass

            output = model.forward(inputs)

            # Calculate Loss

            valloss = criterion(output, labels)

            # Add loss to the validation set's running loss

            val_loss += valloss.item()*inputs.size(0)

            # Since our model outputs a LogSoftmax, find the real 

            # percentages by reversing the log function

            output = torch.exp(output)

            # Get the top class of the output

            top_p, top_class = output.topk(1, dim=1)

            # See how many of the classes were correct?

            equals = top_class == labels.view(*top_class.shape)

            # Calculate the mean (get the accuracy for this batch)

            # and add it to the running accuracy for this epoch

            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # Print the progress of our evaluation

            counter += 1

    # Get the average loss for the entire epoch

    train_loss = train_loss/len(trainloader.dataset)

    train_loss_list.append(train_loss)

    valid_loss = val_loss/len(testloader.dataset)

    validation_loss_list.append(valid_loss)

    train_acc = accuracy/len(trainloader)

    train_accuracy_list.append(train_acc)

    test_acc = accuracy/len(testloader)

    test_accuracy_list.append(test_acc)

    epoch_list.append(epoch)

    print(f'Epoch: {epoch}, Accuracy: {accuracy}, Train loss: {train_loss}, Valid loss: {valid_loss}')
max_train_accuracy = max(train_accuracy_list)

max_test_accuracy = max(test_accuracy_list)

epoch_train = epoch_list[train_accuracy_list.index(max_train_accuracy)]

epoch_test = epoch_list[test_accuracy_list.index(max_test_accuracy)]

print(f'Max Accuracy Score: {max_train_accuracy}, Epoch: {epoch_train}')

print(f'Max Accuracy Score: {max_test_accuracy}, Epoch: {epoch_test}')
epochs = range(1, len(train_accuracy_list) + 1)

#Train and validation accuracy

plt.plot(epochs, train_accuracy_list, 'b', label='Train accurarcy')

plt.plot(epochs, test_accuracy_list, 'r', label='Validation accurarcy')

plt.title('Training and Validation accurarcy')

plt.legend()



plt.figure()

#Train and validation loss

plt.plot(epochs, train_loss_list, 'b', label='Training loss')

plt.plot(epochs, validation_loss_list, 'r', label='Validation loss')

plt.title('Training and Validation loss')

plt.legend()

plt.show()
#Save the model

print("Our model: \n\n", model, '\n')

print("The state dict keys: \n\n", model.state_dict().keys())
torch.save(model.state_dict(), 'checkpoint.pth')
state_dict = torch.load('checkpoint.pth')

print(state_dict.keys())