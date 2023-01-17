import torch

import torch.nn as nn

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from torch.utils import data





# hyper parameters

learning_rate = 0.02

batch_size = 50

num_epochs = 10





# model definition

class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(32*5*5, 10)

        

    def forward(self, input):

        out = self.pool1(self.relu1(self.conv1(input)))

        out = self.pool2(self.relu2(self.conv2(out)))

        out = out.view(-1, 32*5*5)

        out = self.fc(out)

        return out

    

    

# instantiate a model

net = CNN()

    

    

# loss function and optimizer

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)





# data preprocessing

train_data_file = '/kaggle/input/digit-recognizer/train.csv'

test_data_file = '/kaggle/input/digit-recognizer/test.csv'

train_data = pd.read_csv(train_data_file).to_numpy()

test_data = pd.read_csv(test_data_file).to_numpy()

input_data = train_data[:, 1:] / 255.0

label_data = train_data[:, :1]

test_input = test_data / 255.0

train_input, val_input, train_label, val_label = train_test_split(input_data, label_data, test_size=0.2)

train_input = torch.Tensor(train_input)

val_input = torch.Tensor(val_input)

train_label = torch.Tensor(train_label).type(torch.LongTensor).squeeze()

val_label = torch.Tensor(val_label).type(torch.LongTensor).squeeze()

test_input = torch.Tensor(test_input)

train_dataset = data.TensorDataset(train_input, train_label)

val_dataset = data.TensorDataset(val_input, val_label)

test_dataset = data.TensorDataset(test_input)

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)





# train and validate

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        optimizer.zero_grad()

        images = images.view(-1, 1, 28, 28)

        output = net(images)

        loss = loss_function(output, labels)

        loss.backward()

        optimizer.step()

    total = 0

    correct = 0

    for images, labels in val_loader:

        images = images.view(-1, 1, 28, 28)

        output = net(images)

        predictions = torch.max(output, dim=1)[1]

        total += len(labels)

        correct += (predictions == labels).sum()

    print('epoch:%d, total:%d, correct:%d, accuracy:%f.' % (epoch, total, correct, correct/total))

    

    

# test

result = []

for i, image in enumerate(test_loader):

    image = image[0].view(-1, 1, 28, 28)

    output = net(image)

    predictions = torch.max(output, dim=1)[1]

    result.append([i+1, predictions.data[0]])

result = np.array(result, dtype=np.int32)

df = pd.DataFrame(result, columns=["ImageId", "Label"])

df.to_csv("cnn_result.csv", index=False)