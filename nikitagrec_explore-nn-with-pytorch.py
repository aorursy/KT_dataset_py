import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

data.head(3)
fig = plt.figure(figsize=(15,15))

for i in range(1,10):

    fig.add_subplot(3,3,i)

    sns.heatmap(np.array(data.iloc[i-1,1:]).reshape(28,28), cbar=False, cmap='Greys');
import torch

from torch import nn

import torchvision

from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from PIL import Image
mean = np.array(data.iloc[:,1:]).flatten().mean()

std = np.array(data.iloc[:,1:]).flatten().std()

print('mean = ', mean)

print('std = ', std)
class Fashion(Dataset):

    def __init__(self, data, transform=None):

        self.transform = transform

        dataset = data

        self.labels = dataset.label.values

        self.images = dataset.iloc[:, 1:].values.astype('uint8').reshape(-1, 28, 28)



    def __len__(self):

        return len(self.images)



    def __getitem__(self, idx):

        label = self.labels[idx]

        img = Image.fromarray(self.images[idx])

        

        if self.transform:

            img = self.transform(img)



        return img, label
transf = transforms.Compose([transforms.ToTensor(), 

                            transforms.Normalize( (mean/225,), (std/225,) )])



data_train = Fashion(pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv'),transform=transf)

data_test = Fashion(pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv'),transform=transf)



train_loader = torch.utils.data.DataLoader(data_train, batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(data_test, batch_size=64, shuffle=True)
class Net(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):

        super().__init__()

        self.relu = nn.ReLU()

        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(input_size, hidden_size[0])

        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])

        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])

        self.fc4 = nn.Linear(hidden_size[2], num_classes)

    def forward(self, x):

        out = self.fc1(x)

        out = self.relu(out)

        out = self.fc2(out)

        out = self.tanh(out)

        out = self.fc3(out)

        out = self.tanh(out)

        out = self.fc4(out)

        return out
params = []

loss_train = []

loss_test = []



def test(model):

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        correct = 0

        total = 0

        for batch_id, (image, label) in enumerate(test_loader):

            image = image.view(image.shape[0],-1)

            outputs = model(image)

            

            # add loss to graph

            if (batch_id % 100 == 0) & (batch_id > 0):

                loss = criterion(outputs, label)

                loss_test.append(loss.item())

            predicted = torch.argmax(outputs,dim=1)

            total += label.size(0)

            correct += (predicted == label).sum().item()

        print('Test accuracy: {} %'.format(100 * correct / total))

        

def train():

    model = Net(784, [150,150,150], 10)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 20):

        for batch_id, (image, label) in enumerate(train_loader):

            image = image.view(image.shape[0],-1)

            output = model(image)

            loss = criterion(output, label)



            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



            if batch_id % 1024 == 0:

                print('Loss :{:.4f} Epoch - {}/{} '.format(loss.item(), epoch, 20), end=' ')

                loss_train.append(loss.item())

        test(model)

    for i in model.parameters():

        params.append(i)

    return model
model = train()
plt.plot(np.arange(19),loss_train, label='train')

plt.scatter(np.arange(19),loss_test, label = 'test')

plt.legend();
print("Classes - T-shirt/Top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot")

for i in model.parameters():

        params.append(i)

        

fig = plt.figure(figsize=(20,25))

for i in range(1,25):

    fig.add_subplot(7,4,i)

    sns.heatmap(params[0][i-1,:].reshape(28,28).detach().numpy(), cbar=False);
class Net(torch.nn.Module):

    def __init__(self, input_size, num_classes):

        super().__init__()

        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):

        out = self.fc1(x)

        return out

params = []

loss_train = []

loss_test = []



def test(model):

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        correct = 0

        total = 0

        for batch_id, (image, label) in enumerate(test_loader):

            image = image.view(image.shape[0],-1)

            outputs = model(image)

            

            # add loss to graph

            if (batch_id % 100 == 0) & (batch_id > 0):

                loss = criterion(outputs, label)

                loss_test.append(loss.item())

            

            predicted = torch.argmax(outputs,dim=1)

            total += label.size(0)

            correct += (predicted == label).sum().item()

        print('Test accuracy: {} %'.format(100 * correct / total))

        

def train():

    model = Net(784, 10)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 20):

        for batch_id, (image, label) in enumerate(train_loader):

            image = image.view(image.shape[0],-1)

            output = model(image)

            loss = criterion(output, label)



            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



            if batch_id % 1024 == 0:

                print('Loss :{:.4f} Epoch - {}/{}'.format(loss.item(), epoch, 20), end=' ')

                loss_train.append(loss.item())

        test(model)

    for i in model.parameters():

        params.append(i)

    return model

model = train()
plt.plot(np.arange(19),loss_train, label='train')

plt.scatter(np.arange(19),loss_test, label = 'test')

plt.legend();
print("Classes - T-shirt/Top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot")

for i in model.parameters():

        params.append(i) 

fig = plt.figure(figsize=(15,15))

for i in range(1,11):

    fig.add_subplot(5,4,i)

    sns.heatmap(params[0][i-1,:].reshape(28,28).detach().numpy(), cbar=False);
nb_classes = 10

confusion_matrix = torch.zeros(nb_classes, nb_classes)

with torch.no_grad():

    for i,  (image, classes) in enumerate(test_loader):

        image = image.view(image.shape[0],-1)

        outputs = model(image)

        _, preds = torch.max(outputs, dim=1)

        for t, p in zip(classes.view(-1), preds.view(-1)):

                confusion_matrix[t.long(), p.long()] += 1

cols = ["T-shirt/Top", "Trouser", "Pullover","Dress","Coat", "Sandal", "Shirt","Sneaker","Bag","Ankle Boot"]

plt.figure(figsize=(10,8))

sns.heatmap(pd.DataFrame(np.array(confusion_matrix), columns=cols, index=cols), cmap='Greys', annot=True);