import random



import matplotlib.pyplot as plt

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

import torch.cuda as cuda

from albumentations import (HorizontalFlip, Compose)

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset
def visualize(image, label):

    plt.imshow(image, interpolation='nearest')

    plt.title('There are digit {} on this picture'.format(label))

    plt.show()

    

def get_transforms():

    return Compose[HorizontalFlip(p=0.5)]



def train(model, device, train_loader, epoch):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = F.cross_entropy(output, target)

        loss.backward()

        optimizer.step()

        if batch_idx % 1000 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, batch_idx * len(data), len(train_loader.dataset),

                100. * batch_idx / len(train_loader), loss.item()))

            

def test(model, device, test_loader):

    model.eval()

    average_loss = 0

    correct = 0

    with torch.no_grad():

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            average_loss += F.cross_entropy(output, target)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            correct += int(pred.eq(target.view_as(pred)).sum())

    average_loss /= len(test_loader.dataset)

    

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

        average_loss, correct, len(test_loader.dataset),

        100. * correct / len(test_loader.dataset)))

train_file_path = '../input/digit-recognizer/train.csv'

train_images = pd.read_csv(train_file_path)





train_images['Image'] = train_images.iloc[:, 1:].apply(lambda x: x.values.reshape(28, 28), axis=1)

train_images = train_images.drop(train_images.columns[1:785], axis=1)



train_ids, valid_ids = train_test_split(train_images.index.values, random_state=42, test_size=0.1)



train_img = train_images.iloc[random.choice(train_ids)]

valid_img = train_images.iloc[random.choice(valid_ids)]

visualize(train_img['Image'], train_img['label'])

visualize(valid_img['Image'], valid_img['label'])
class MnistDataset(Dataset):

    def __init__(self, df):

        self.df = df

        # self.transforms = get_transforms()

        super().__init__()

    

    def __getitem__(self, index: int):

        return np.expand_dims(self.df.iloc[index]['Image'], axis=0).astype(np.float32), self.df.iloc[index]['label']

    

    def __len__(self):

        return len(self.df)

    

class Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv_part = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.BatchNorm2d(16),

            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1, bias=False),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.BatchNorm2d(64)

        )

        self.linear_part = nn.Sequential(

            nn.Linear(3136, 50),

            nn.Dropout(0.5),

            nn.Linear(50, 10)

        )

    

    def forward(self, x):

        out = self.conv_part(x)

        out = torch.flatten(out, 1)

        out = self.linear_part(out)

        return F.log_softmax(out)


batch_size = 50

num_workers = 4

num_epochs = 12



train_dataset = MnistDataset(train_images.iloc[train_ids])

test_dataset = MnistDataset(train_images.iloc[valid_ids])

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
model = Model()



device = 'cpu'

if cuda.is_available:

    device = 'cuda'

    model.cuda()

    

optimizer = torch.optim.Adam([

    {'params': model.parameters(), 'lr': 1e-3},

])

    

for epoch in range(num_epochs):

    train(model, device, train_loader, epoch)

    test(model, device, test_loader)


test_file_path = '../input/digit-recognizer/test.csv'

test_images = pd.read_csv(test_file_path)





test_images['Image'] = test_images.apply(lambda x: x.values.reshape(28, 28), axis=1)

test_images = test_images.drop(test_images.columns[:784], axis=1)



test_dataset = MnistDataset(train_images)



visualize(test_images.iloc[0]['Image'], 'something')



model.eval()

submission = pd.DataFrame(data={'ImageId': [], 'Label': []})

with torch.no_grad():

    for idx, elem in enumerate(test_images['Image']):

        elem = np.expand_dims(elem, axis=0)

        elem = np.expand_dims(elem, axis=0)

        data = torch.from_numpy(elem.astype('float32'))

        data = data.to('cuda')

        output = model(data)

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

        submission = submission.append(pd.DataFrame.from_records([{'ImageId': idx+1, 'Label': int(pred)}]))

submission.astype(int).to_csv("cnn_mnist_datagen.csv",index=False)


