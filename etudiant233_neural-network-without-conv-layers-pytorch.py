import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import math

import random

import os

from tqdm import tqdm



import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, random_split
def encode_onehot(labels, size=10):

    """

    One-hot encoding

    """

    encoded = []

    for label in labels:

        encoded.append(np.eye(size)[int(label)])

    return encoded



def normalize(img):

    """

    Normalize the image pixel values

    """

    img = np.array(img)

    img = (img - 128) / 128

    return img



class MNISTDataset(Dataset):

    def __init__(self, csv_file, is_test=False):

        self.is_test = is_test

        

        df = pd.read_csv(csv_file)

        

        if not self.is_test:

            self.labels = df['label'].to_numpy(dtype='int32')

            self.imgs = df.to_numpy()[:, 1:]

        else:

            self.imgs = df.to_numpy()

        

        for i in range(len(self.imgs)):

            self.imgs[i] = normalize(self.imgs[i])

        

        if not self.is_test:

            self.labels = torch.Tensor(self.labels).type(torch.long)

        

        self.imgs = torch.Tensor(self.imgs)



    def __len__(self):

        return len(self.imgs)

    

    def __getitem__(self, idx):

        if self.is_test:

            return self.imgs[idx]

        else:

            return self.labels[idx], self.imgs[idx]



def plot_dataset(ds):

    """

    Plot 10 sample images from the dataset

    """

    nsamples = 10

    ids = random.sample(range(len(ds)), nsamples)

    fig, axes = plt.subplots(1, 10, figsize=(20, 200))

    axes = axes.ravel()

    size = 28 # the image size is 27x27

    for idx, ax in zip(ids, axes):

        label, img = ds[idx]

        img = img.reshape((size, size))

        ax.matshow(img)

        ax.set_title(f'{label}')

        ax.axis('off')
ds = MNISTDataset(csv_file='../input/digit-recognizer/train.csv')



plot_dataset(ds)
train_size = int(0.8 * len(ds))

val_size = len(ds) - train_size



train_set, val_set = random_split(ds, [train_size, val_size])
batch_size = 128

num_workers = 2



train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)

val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
test_set = MNISTDataset(csv_file='../input/digit-recognizer/test.csv', is_test=True)



for img in test_set:

    plt.matshow(img.reshape((28, 28)))

    break
class MLPNetwork(nn.Module):

    def __init__(self, in_dim, out_dim=10):

        super().__init__()

        self.fc1 = nn.Linear(in_dim, 1024)

        self.drop1 = nn.Dropout(p=0.5)

        #self.fc2 = nn.Linear(128, 128)

        #self.drop2 = nn.Dropout(p=0.25)

        #self.fc3 = nn.Linear(32, 32)

        #self.fc4 = nn.Linear(32, 16)

        self.fc2 = nn.Linear(1024, out_dim)

    

    def forward(self, x):

        x = self.fc1(x)

        x = F.relu(x)

        x = self.drop1(x)

        x = self.fc2(x)

        #x = F.relu(x)

        #x = self.drop2(x)

        #x = self.fc3(x)

        #x = F.relu(x)

        #x = self.fc4(x)

        #x = F.relu(x)

        #x = self.fc5(x)

        x = F.softmax(x, dim=-1)

        return x
def train(model, criterion, optimizer, train_loader, val_loader, max_epochs=100, device='cpu', output_dir='useless'):

    os.makedirs(output_dir, exist_ok=True)

    

    history = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])

    

    model = model.to(device)

    for epoch in range(max_epochs):

        model.train()

        train_loss = 0



        for label, data in train_loader:

            label, data = label.to(device), data.to(device)

            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, label)

            train_loss += loss

            loss.backward()

            optimizer.step()

        train_loss /= len(train_loader.dataset)



        model.eval()

        val_loss = 0

        val_correct = 0

        train_corret = 0

        with torch.no_grad():

            for label, data in train_loader:

                label, data = label.to(device), data.to(device)

                output = model(data)

                pred_label = torch.argmax(output, dim=1)

                train_corret += float(sum(pred_label == label))



            for label, data in val_loader:

                label, data = label.to(device), data.to(device)

                output = model(data)

                loss = criterion(output, label)

                val_loss += loss

                pred_label = torch.argmax(output, dim=1)

                val_correct += float(sum(pred_label == label))

            val_loss /= len(val_loader.dataset)

        val_acc = val_correct / len(val_loader.dataset)

        train_acc = train_corret / len(train_loader.dataset)

        print(f"[epoch {epoch}/{max_epochs}] train loss: {train_loss:.5f} train acc: {train_acc:.5f} "

              f"val loss: {val_loss:.5f} val acc: {val_acc:.5f}")

        history = history.append({'epoch':epoch, 'train_loss':train_loss, 'val_loss':val_loss, 'train_acc':train_acc, 'val_acc':val_acc}, ignore_index=True)

        torch.save(model, os.path.join(output_dir, f'epoch.{epoch}.pt'))

    

    history.to_csv(os.path.join(output_dir, 'history.csv'))    

    return {'model':model, 'history':history}

def plot_history(hist, filename=None):

    if len(hist) == 0:

        raise ValueError("Cannot plot empty history!")

    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    axes[0].plot(hist.epoch, hist.train_loss, label='train loss')

    axes[0].plot(hist.epoch, hist.val_loss, label='val loss')

    axes[0].legend()

    axes[0].set_title('Loss')

    axes[1].plot(hist.epoch, hist.train_acc, label='train acc')

    axes[1].plot(hist.epoch, hist.val_acc, label='val acc')

    axes[1].legend()

    axes[1].set_title('Accuracy')

    

    if filename:

        fig.savefig(filename)
IMGSIZE = 28

mlp_net = MLPNetwork(IMGSIZE * IMGSIZE, 10)

print(mlp_net)



criterion = nn.CrossEntropyLoss()

learning_rate = 1e-3

optimizer = optim.Adam(mlp_net.parameters(), lr=learning_rate)



result = train(model=mlp_net, 

              criterion=criterion, 

              optimizer=optimizer, 

              train_loader=train_loader, 

              val_loader=val_loader, 

              max_epochs=100, 

              device='cpu',

              output_dir=f'linear1024-drop05')

plot_history(result['history'], filename=f'linear2048-drop05.png')
submission_df = pd.DataFrame(columns=['ImageId', 'Label'])

model = result['model']

for idx, img in enumerate(tqdm(test_set), start=1):

    with torch.no_grad():

        output = model(img)

        output = int(torch.argmax(output))

        submission_df = submission_df.append({'ImageId':idx, 'Label':output}, ignore_index=True)



submission_df.head()
submission_df.to_csv('submission.csv', index=False)