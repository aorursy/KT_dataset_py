import torch

import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
# Hyperparameters

num_epochs = 6

num_classes = 10

batch_size = 100

learning_rate = 0.01
train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv", dtype=np.float32)

test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv", dtype=np.float32)

sample_sub = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
train_df.shape
y = train_df.label.values

X0 = train_df.loc[:, train_df.columns != 'label'].values/255



X0.shape, y.shape
X = X0.reshape((-1, 1, 28, 28))



X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



X_train.shape, X_test.shape
XTrain = torch.from_numpy(X_train)

yTrain = torch.from_numpy(y_train).type(torch.LongTensor)



XTest = torch.from_numpy(X_test)

yTest = torch.from_numpy(y_test).type(torch.LongTensor)
train = TensorDataset(XTrain, yTrain)

test = TensorDataset(XTest, yTest)



train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)

test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)
class ConvNet(nn.Module):

    def __init__(self):

        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout()

        self.fc1 = nn.Linear(7 * 7 * 64, 1000)

        self.fc2 = nn.Linear(1000, 10)



    def forward(self, x):

        out = self.layer1(x)

        out = self.layer2(out)

        out = out.reshape(out.size(0), -1)

        out = self.drop_out(out)

        out = self.fc1(out)

        out = self.fc2(out)

        return out





model = ConvNet()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(train_loader)

loss_list = []

acc_list = []

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        # Run the forward pass

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss_list.append(loss.item())



        # Backprop and perform Adam optimisation

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        # Track the accuracy

        total = labels.size(0)

        _, predicted = torch.max(outputs.data, 1)

        correct = (predicted == labels).sum().item()

        acc_list.append(correct / total)



        if (i + 1) % 100 == 0:

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'

                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),

                          (correct / total) * 100))
# Test the model

model.eval()

with torch.no_grad():

    correct = 0

    total = 0

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



    print('Accuracy of the model on test images: {} %'.format((correct / total) * 100))
Xt = (test_df.values/255).reshape((-1, 1, 28, 28))

Xt.shape
model.eval()

with torch.no_grad():

    sample_sub['Label'] = model(torch.from_numpy(Xt)).argmax(dim=1)

    

sample_sub.head()
sample_sub.to_csv("submission.csv", index=False)