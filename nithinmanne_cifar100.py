# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np



dataset_prefix = "cifar100" 

num_classes = 100



train_data = np.load("/kaggle/input/2020-mpcs53111-hw5-cifar100/{}_train.npy".format(dataset_prefix))

test_data = np.load("/kaggle/input/2020-mpcs53111-hw5-cifar100/{}_test.npy".format(dataset_prefix))



train_images = train_data[:, :-1].reshape(-1, 3, 32, 32)

train_labels = train_data[:, -1]

test_images = test_data.reshape(-1, 3, 32, 32)
import torchvision

from PIL import Image



pil_images = [Image.fromarray((np.moveaxis(image, (0,1,2), (2, 0, 1))*255).astype("uint8"), 'RGB') for image in train_images]

def pil_to_np(images):

    images = np.array([np.array(image) for image in images])

    return np.moveaxis(images, (1, 2, 3), (2, 3, 1))/255.

modified_images = [torchvision.transforms.functional.hflip(image) for image in pil_images]

train_images = np.concatenate([train_images, pil_to_np(modified_images)], axis=0)

modified_images = [torchvision.transforms.functional.affine(image, 0, (-2, 0), 1, 0) for image in pil_images]

train_images = np.concatenate([train_images, pil_to_np(modified_images)], axis=0)

modified_images = [torchvision.transforms.functional.affine(image, 0, (+2, 0), 1, 0) for image in pil_images]

train_images = np.concatenate([train_images, pil_to_np(modified_images)], axis=0)

#modified_images = [torchvision.transforms.functional.affine(image, 0, (0, -2), 1, 0) for image in pil_images]

#train_images = np.concatenate([train_images, pil_to_np(modified_images)], axis=0)

#modified_images = [torchvision.transforms.functional.affine(image, 0, (0, +2), 1, 0) for image in pil_images]

#train_images = np.concatenate([train_images, pil_to_np(modified_images)], axis=0)

modified_images = None

pil_images = None

train_labels = np.concatenate([train_labels]*4, axis=0)
import torch

import torch.nn as nn

import torch.optim as optim

import torch.utils as utils

import torch.nn.functional as F



class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn_layers = nn.Sequential(

            nn.Conv2d(3, 3*64, 3, padding=1),

            nn.ReLU(),

            nn.Conv2d(3*64, 3*128, 3, padding=1),

            nn.ReLU(),

            nn.BatchNorm2d(3*128),

            nn.MaxPool2d(2, 2),

            nn.Dropout2d(p=0.2),

            nn.Conv2d(3*128, 3*128, 3, padding=1),

            nn.ReLU(),

            nn.Conv2d(3*128, 3*256, 3, padding=1),

            nn.ReLU(),

            nn.BatchNorm2d(3*256),

            nn.MaxPool2d(2, 2),

            nn.Dropout2d(p=0.2),

            nn.Conv2d(3*256, 3*256, 3, padding=1),

            nn.ReLU(),

            nn.Conv2d(3*256, 3*256, 3, padding=1),

            nn.ReLU(),

            nn.Conv2d(3*256, 3*512, 3, padding=1),

            nn.ReLU(),

            nn.BatchNorm2d(3*512),

            nn.MaxPool2d(2, 2),

            nn.Dropout2d(p=0.2),

        )

        self.linear_layers = nn.Sequential(

            nn.Linear(24576, 2048),

            nn.ReLU(),

            nn.Dropout(p=0.2),

            nn.Linear(2048, 768),

            nn.ReLU(),

            nn.Linear(768, 100),

        )



    def forward(self, x):

        x1 = self.cnn_layers(x).view(x.shape[0], -1)

        return self.linear_layers(x1)



class Model:

    def fit(self, X, y, lr=0.001, n_epochs=50, test_data=None):

        self.net = Net()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.net = self.net.to(self.device)

        self.loss = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        train_data = (X, y)

        X = torch.from_numpy(X).float().to(self.device)

        y = torch.from_numpy(y).long().to(self.device)

        data = utils.data.TensorDataset(X, y)

        data_loader = utils.data.DataLoader(data, batch_size=64, shuffle=True)

        for ep in range(n_epochs):

            self.net.train()

            for batch_X, batch_y in data_loader:

                self.optimizer.zero_grad()

                y_hat = self.net(batch_X)

                loss = self.loss(y_hat, batch_y)

                loss.backward()

                self.optimizer.step()

            if ep % 1 == 0:

                print(f'Epoch: {ep}, Loss: {loss}')

                print(f'Train Accuracy: {self.score(*train_data)}')

                if test_data: print(f'Test Accuracy: {self.score(*test_data)}')

    def predict(self, X):

        with torch.no_grad():

            self.net.eval()

            X = torch.from_numpy(X).float().to(self.device)

            data = utils.data.TensorDataset(X)

            data_loader = utils.data.DataLoader(data, batch_size=64)

            outputs = []

            for batch, in data_loader:

                outputs.append(self.net(batch))

            out = torch.cat(outputs)

            return torch.argmax(out, axis=1)

    def score(self, X, y):

        y_hat = self.predict(X)

        y = torch.from_numpy(y).long().to(self.device)

        return int((y == y_hat).sum())/y.shape[0]

    def get_params(self, deep=True): return {}

    def set_params(self, **__): pass
model = Model()

model.fit(train_images, train_labels)
out = model.predict(test_images)

import csv

with open(f'{dataset_prefix}_out.csv', mode='w', newline='') as out_file:

    fieldnames = ['Id', 'Category']

    writer = csv.DictWriter(out_file, fieldnames=fieldnames)



    writer.writeheader()

    for i, j in enumerate(out):

        writer.writerow({'Id': i, 'Category': int(j)})