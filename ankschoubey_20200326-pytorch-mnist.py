import torch

import numpy as np

import pandas as pd



train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

sample_df = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
from torch.utils.data.dataset import Dataset



class MnistDataset(Dataset):

    def __init__(self, df, transforms=None, train=True):

        self.train = train

        self.transforms = transforms

        

        self.X = df.loc[:, df.columns != 'label'].to_numpy(float)

        self.X = torch.from_numpy(self.X)

        if train:

            self.y = df.get('label').to_numpy()

            self.y = torch.from_numpy(self.y)

    

    def __len__(self):

        return self.X.shape[0]

    

    def __getitem__(self, i):

        x = self.X[i].view(1, 28, 28).expand(3, 28, 28)

        

        if self.transforms: x = self.transforms(x)

        

        if self.train: return x, self.y[i]

        return x



#Convert GrayScale to RGB: https://discuss.pytorch.org/t/grayscale-to-rgb-transform/18315/7
#https://pytorch.org/docs/stable/torchvision/models.html

import  torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225])
ds = MnistDataset(train_df, normalize)

len(ds)
val_len = int(len(ds)*0.01) # 0.01 percent of data

train_len = len(ds) - val_len # all other are in training



from torch.utils.data import random_split

train_ds, val_ds = random_split(ds, [train_len, val_len])



len(train_ds), len(val_ds)
from torch.utils.data.dataloader import DataLoader



bs = 512

num_workers = 2

train_dl = DataLoader(train_ds, bs, num_workers=num_workers)

val_dl = DataLoader(val_ds, bs, num_workers=num_workers)
images, labels = next(iter(train_dl))

images.shape, labels.shape
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)



resnet18.fc
lin_in = resnet18.fc.in_features



import torch.nn as nn



resnet18.fc = nn.Sequential(

    nn.Linear(lin_in, 10)

)
net = resnet18

out = net(images.float())

out.shape
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

net = net.to(device)

device
epochs = 10

loss_fn = nn.CrossEntropyLoss()



import torch.optim as optim

o = optim.Adam(net.parameters(),lr=0.001)
from datetime import datetime



losses = []

accuracies = []

for i in range(epochs):

    e_loss = 0

    start = datetime.now()

    # training

    for images, labels in train_dl:

        o.zero_grad()

        

        images, labels = images.to(device,dtype=torch.float), labels.to(device)

        

        out = net(images)

        loss = loss_fn(out.float(), labels.long())

        loss.backward()

        

        e_loss += loss.item()

        

        o.step()



    #validation

    with torch.no_grad():

        accuracy = 0

        for images, labels in val_dl:

            images, labels = images.to(device,dtype=torch.float), labels.to(device)

            out = net(images)

            accuracy+=(out.argmax(dim=1) == labels).sum().item()

        accuracies.append(accuracy/len(val_ds) * 100)



    end = datetime.now()



    print(f'Epoch: {i}\tTime: {(end - start).total_seconds():.2f}s\tLoss: {e_loss:.2f}\tAccuracy: {accuracies[-1]:.2f}')

    losses.append(e_loss)
import matplotlib.pyplot as plt



plt.plot(range(epochs), accuracies)
bs = 640

test_ds = MnistDataset(test_df, normalize, train = False)

test_dl = DataLoader(test_ds, bs)
outputs = []

with torch.no_grad():

    for images in test_dl:

        images = images.to(device, dtype=torch.float)

        out = net(images)

        outputs.extend(out.argmax(dim=1).tolist())
len(outputs)
test_df.shape, sample_df.shape
sample_df['Label'] = outputs

sample_df.to_csv('submission.csv', index=False)