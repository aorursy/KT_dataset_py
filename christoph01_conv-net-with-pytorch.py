import torch

import torch.nn as nn

import torch.nn.functional as f

import torch.utils.data as data_utils

from torch.utils.data.dataset import Dataset

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from torch.utils import data

import pandas as pd

from PIL import Image



import os

print(os.listdir("../input"))
class FashionMNIST(Dataset):

    def __init__(self, path, transform=None):

        self.transform = transform

        df = pd.read_csv(path)

        self.labels = df.label.values

        self.images = df.iloc[:, 1:].values.astype("uint8").reshape(-1, 28, 28)



    def __len__(self):

        return len(self.images)



    def __getitem__(self, idx):

        y = self.labels[idx]

        X = Image.fromarray(self.images[idx])

        

        if self.transform:

            X = self.transform(X)



        return X, y
transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(5), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_data = FashionMNIST("../input/fashion-mnist_train.csv", transform)



transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

test_data = FashionMNIST("../input/fashion-mnist_test.csv", transform)



batch_size = 150



train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv_block = nn.Sequential(nn.Conv2d(1, 128, 3, 1, 1),

                                   nn.ReLU(),

                                   nn.MaxPool2d(2, 2),

                                   nn.Conv2d(128, 256, 3, 1, 1),

                                   nn.ReLU(),

                                   nn.MaxPool2d(2, 2))



        self.clf = nn.Sequential(nn.Linear(7 * 7 * 256, 1024),

                                  nn.ReLU(),

                                  nn.Linear(1024, 10),

                                  nn.Softmax(dim=1))



    def forward(self, x):

        x = self.conv_block(x)

        x = x.view(-1, 7 * 7 * 256)

        return self.clf(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)



model = Net()



model.to(device)

print(model)



#define loss

criterion = nn.CrossEntropyLoss()



#define optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
acc_hist_train = []

acc_hist_test = []

n_epochs = 30



for epoch in range(n_epochs):

    #set train mode

    model.train() 

    acc_train = []

    acc_test = []

        

    for X,y in train_loader:

        X = X.to(device)

        y = y.to(device)

        optimizer.zero_grad()

        out = model(X)

        

        pred = out.detach().cpu().numpy()

        label = y.detach().cpu().numpy()

        a = (pred.argmax(axis=1) == label)

        acc_train.extend(a)

        

        loss = criterion(out, y)

        loss.backward()

        optimizer.step()

        

      

    print("Training Accuracy for {}: {}%".format(epoch+1, sum(acc_train) / len(acc_train) * 100))

    acc_hist_train.append(sum(acc_train) / len(acc_train))

        

    model.eval()

    

    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(device)

            y = y.to(device)

            out = model(X)

            

            pred = out.detach().cpu().numpy()

            label = y.detach().cpu().numpy()

            a = (pred.argmax(axis=1) == label)

            acc_test.extend(a)

            

            loss = criterion(out, y)

            

        print("Validation Accuracy for {}: {}%".format(epoch+1, sum(acc_test) / len(acc_test) * 100))

        acc_hist_test.append(sum(acc_test) / len(acc_test))

plt.plot(acc_hist_train, label="Training")

plt.plot(acc_hist_test, label="Test")

plt.legend(loc="lower right")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.show()