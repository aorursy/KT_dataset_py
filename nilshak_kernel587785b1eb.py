!pip install jovian --upgrade --quiet
import torch

import torch.nn as nn

from torchvision import datasets

import torchvision.transforms as transforms

from torch.autograd import Variable



# Hyperparameters

input_size = 784  # Our images are 28px by 28px in size

num_classes = 10  # We have handwritten digits from 0 - 9

num_epochs = 5  # Number of epochs

batch_size = 100  # Batch size

learning_rate = 0.001  # Learning rate



transfm = transforms.ToTensor()  # Transform the dataset objects to tensors



# MNIST dataset - images and labels

train_dataset = datasets.MNIST(root='./data',

                               train=True,

                               transform=transfm,

                               download=True)



test_dataset = datasets.MNIST(root='./data',

                              train=False,

                              transform=transfm)



# Input pipeline

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,

                                           batch_size=batch_size,

                                           shuffle=True)



test_loader = torch.utils.data.DataLoader(dataset=test_dataset,

                                          batch_size=batch_size,

                                          shuffle=True)

class LogisticRegression(nn.Module):

    def __init__(self, input_size, num_classes):

        super().__init__()

        self.linear = nn.Linear(input_size, num_classes)

        

    def forward(self, x):

        y_hat = self.linear(x)

        return y_hat
model = LogisticRegression(input_size, num_classes)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Training the model

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        images = Variable(images.view(-1, 28 * 28))  # Images flattened into 1D tensors

        labels = Variable(labels)  # Labels 

        

        # Forward -> Backprop -> Optimize

        optimizer.zero_grad()  # Manually zero the gradient buffers

        outputs = model(images)  # Predict the class using the test set

        loss = criterion(outputs, labels)  # Compute the loss given the predicted label

                                           # and actual label

        

        loss.backward()  # Compute the error gradients

        optimizer.step()  # Optimize the model via Stochastic Gradient Descent

        

        if (i + 1) % 100 == 0:

            print("Epoch {}, loss :{}".format(epoch + 1, loss.data[0]))
# Test the Model

correct = 0

total = 0

for images, labels in test_loader:

    images = Variable(images.view(-1, 28 * 28))

    outputs = model(images)

    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)

    correct += (predicted == labels).sum()

 

print('Accuracy: {}%'.format(100 * correct / total))