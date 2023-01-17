# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
! cp -a /kaggle/input/catalyst/catalyst/catalyst/install.sh /tmp/install.sh && chmod 777 /tmp/install.sh && /tmp/install.sh /kaggle/input/catalyst/catalyst/catalyst
import collections

import torch

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

import torch

import torch.nn as nn

import torch.nn.functional as F

from catalyst.dl import utils

from catalyst.dl.runner import SupervisedRunner

from catalyst.dl.callbacks import EarlyStoppingCallback, AccuracyCallback

from catalyst.contrib.schedulers import OneCycleLR
from pathlib import Path
mnist = Path("/kaggle/input/Kannada-MNIST")
train_df = pd.read_csv(mnist/"train.csv")

train,val = train_test_split(train_df)
test_df = pd.read_csv(mnist/"test.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
class KannadaMnist(Dataset):

    """

    Infer from standard PyTorch Dataset class

    Such datasets are often very useful

    """

    

    def __init__(self,

                 images,

                 labels=None,

                 transform=None,

                ):

        self.X = images

        self.y = labels

        

        self.transform = transform

    

    def __len__(self):

        return len(self.X)



    def __getitem__(self, idx):

        img = np.array(self.X.iloc[idx, :], dtype='uint8').reshape([28, 28, 1])

        if self.transform is not None:

            img = self.transform(img)

        

        if self.y is not None:

            target = self.y.iloc[idx]

            return img, target

        else:

            return img
bs = 32

num_workers = 4





data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])



loaders = collections.OrderedDict()



trainset = KannadaMnist(train.iloc[:,1:],train.iloc[:,0], transform=data_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,shuffle=True, num_workers=num_workers)



testset = KannadaMnist(val.iloc[:,1:],val.iloc[:,0], transform=data_transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=bs,shuffle=False, num_workers=num_workers)



loaders["train"] = trainloader

loaders["valid"] = testloader
class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 10)



    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        # print(x.shape)

        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        # print(x.shape)

        return x
NUM_EPOCHS = 10
# experiment setup



num_epochs = NUM_EPOCHS

logdir = "./logs/mnist_kannada"



# model, criterion, optimizer

model = Net()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters())



# model runner

runner = SupervisedRunner()



# model training

runner.train(

    model=model,

    criterion=criterion,

    optimizer=optimizer,

    loaders=loaders,

    logdir=logdir,

    num_epochs=num_epochs,

    verbose=True

)
test_data  = KannadaMnist(test_df.iloc[:,1:],test_df.iloc[:,0], transform=data_transform)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,shuffle=False, num_workers=num_workers)
runner_out = runner.predict_loader(test_loader, resume=f"{logdir}/checkpoints/best.pth")
!ls logs/mnist_kannada/checkpoints
runner_out.shape
_,v = torch.topk(torch.from_numpy(runner_out),1)
submission = pd.read_csv(mnist/"sample_submission.csv")
submission.columns 
v.numpy().shape
sub = pd.DataFrame({'id':submission.id,'label':np.squeeze(v.numpy())},columns=['id','label'])
sub.to_csv("submission.csv",index=False)