# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import torch 

import torch.nn as nn 

from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import torch.optim as optim
seed = 2

train_dataset = pd.read_csv("../input/train.csv")

train_dataset, test_dataset = train_test_split(train_dataset, test_size=0.1, random_state=seed)

dataset = {

    'train': train_dataset,

    'test': test_dataset

}

print(train_dataset.head(), len(train_dataset))
data_transforms = {

    'train': transforms.Compose([

#         transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(mean=(0.5,), std=(0.5,))

    ]),

    'test': transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(mean=(0.5,), std=(0.5,))

    ])

}
class MnistDataset(Dataset):

    def __init__(self, dataset, transforms):

        self.dataset = list(dataset.values)

        self.transforms = transforms

        img, label = [], []

        for line in self.dataset:

            img.append(line[1:])

            label.append(line[0])

        self.img = np.asarray(img).reshape(dataset.shape[0], 28, 28, 1).astype("float32")

        self.label = np.asarray(label)

    def __getitem__(self, index):

        image = self.img[index]

        label = self.label[index]

        if self.transforms is not None:

            image = self.transforms(image)

        return image, label

    def __len__(self):

        return len(self.label)
num_classes = 10

batch_size = 128

epochs = 20
train_data = MnistDataset(train_dataset, data_transforms['train'])

test_data = MnistDataset(test_dataset, data_transforms['test'])

dataloaders = {

    'train': DataLoader(dataset=train_data, batch_size=batch_size),

    'test': DataLoader(dataset=test_data, batch_size=batch_size)

}
class MyModel(nn.Module):

    def __init__(self, num_classes=10):

        super(MyModel, self).__init__()

        self.layer1 = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(7*7*32, num_classes)

        

    def forward(self, x):

        out = self.layer1(x)

        out = self.layer2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc(out)

        return out
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# img, label = next(iter(dataloaders['train']))

# model = MyModel(num_classes)

# outputs = model(img)

# _, pred = torch.max(outputs, 1)

# # print(pred.shape)

# co = torch.sum(pred == label)

# print(pred, label, pred.shape, label.shape)

# print(co)

# # print(label.shape, type(label))

# print(len(dataloaders['train']))
def train_model():

    model = MyModel(num_classes)

    model.to(device)

    

    best_acc = 0.0

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):

        print("epoch{}/{}".format(epoch+1, epochs))

        print("*"*10)

        for phase in ['train', 'test']:

            if phase == 'train':

                model.train()

            else:

                model.eval()

                

            running_correct = 0

            running_loss = 0.0

            for i, (images, labels) in enumerate(dataloaders[phase]):

                images = images.to(device)

                labels = labels.to(device)

                

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(images)

                    _, pred = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()

                    

                running_loss += loss.item() * images.size(0)

                running_correct += torch.sum(pred == labels)

#             print("{} {}".format(epoch, running_correct))

            epoch_loss = running_loss / len(dataset[phase])

            epoch_acc = running_correct.double() / len(dataset[phase])

            

            print("{} Loss:{:.4f} Acc:{:.4f}".format(phase, epoch_loss, epoch_acc))

            

            if phase == 'test' and epoch_acc > best_acc:

                best_acc = epoch_acc

                

    print("best test acc:{:.4f}".format(best_acc))

train_model()