# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

!pip install d2l==0.13.2 -f https://d2l.ai/whl.html # installing d2l

from d2l import torch as d2l

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import os

import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision.transforms import Normalize

from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

device



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
batch_size = 256

image_size = 72

Train_df = pd.read_csv("/kaggle/input/shopee-product-detection-dataset/shopee-product-detection-dataset/train.csv")

test_df = pd.read_csv("/kaggle/input/shopee-product-detection-dataset/shopee-product-detection-dataset/test.csv")

DATADIR = "/kaggle/input/shopee-product-detection-dataset/shopee-product-detection-dataset/train/train"

total_examples = len(Train_df)
mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]

normalize_transform = Normalize(mean, std)



class Unnormalize:

    """Converts an image tensor that was previously Normalize'd

    back to an image with pixels in the range [0, 1]."""

    def __init__(self, mean, std):

        self.mean = mean

        self.std = std



    def __call__(self, tensor):

        mean = torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)

        std = torch.as_tensor(self.std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)

        return torch.clamp(tensor*std + mean, 0., 1.)

unnormalize_transform = Unnormalize(mean, std)

def make_splits(Dir, metadata_df, frac):

    # Make a validation split. Sample a percentage of the real videos, 

    # and also grab the corresponding fake videos.

   

    val_df = metadata_df.sample(frac=frac, random_state=666)

    # The training split is the remaining videos.

    train_df = metadata_df.loc[~metadata_df.index.isin(val_df.index)]

    val_df.reset_index(drop=True, inplace=True)

    train_df.reset_index(drop=True, inplace=True)

    return train_df, val_df
def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):

    h, w = img.shape[:2]

    if w > h:

        h = h * size // w

        w = size

    else:

        w = w * size // h

        h = size



    resized = cv2.resize(img, (w, h), interpolation=resample)

    return resized





def make_square_image(img):

    h, w = img.shape[:2]

    size = max(h, w)

    t = 0

    b = size - h

    l = 0

    r = size - w

    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)





def load_image_and_label(filename, label):

    if label<10:

        path = os.path.join(DATADIR, '0'+str(label), filename)

    else:

        path = os.path.join(DATADIR, str(label), filename)

    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = isotropically_resize_image(img, 224, resample=cv2.INTER_AREA)

    img = make_square_image(img)

    img = torch.tensor(img).permute((2,0,1)).float().div(255)

    img = normalize_transform(img)

    return img, label

    

        

class Data_set(Dataset):

    def __init__(self,df):

        self.df = df

    

    def __getitem__(self, index):

        filename = self.df['filename'][index]

        label = self.df['category'][index]

        #print("nice")

        return load_image_and_label(filename, label)

    def __len__(self):

        return len(self.df)

        

train_df, val_df = make_splits(DATADIR, Train_df, 0.3)

train_dataset = Data_set(train_df)

val_dataset = Data_set(val_df)
from torch.utils.data import DataLoader

train_iter = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True)

val_iter = DataLoader(val_dataset, batch_size=256, shuffle=True, pin_memory=True)

n = 0

'''for X, y in train_loader:

    n +=1

    print(y)

    if n > 100: break'''
net = nn.Sequential(

    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),

    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),

    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),

    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),

    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),

    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Flatten(),

    nn.Dropout(p=0.5),

    nn.Linear(6400, 4096), nn.ReLU(),

    nn.Dropout(p=0.5),

    nn.Linear(4096, 4096), nn.ReLU(),

    nn.Linear(4096, 42))

net.load_state_dict(torch.load('/kaggle/input/productdetectionparams1/saved_para.pth'))
def train_ch6(net, train_iter, test_iter, num_epochs, lr,

              device=d2l.try_gpu()):

    """Train and evaluate a model with CPU or GPU."""

    '''def init_weights(m):

        if type(m) == nn.Linear or type(m) == nn.Conv2d:

            torch.nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)'''

    print('training on', device)

    net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.8)

    loss = nn.CrossEntropyLoss()

    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],

                            legend=['train loss', 'train acc', 'test acc'])

    timer = d2l.Timer()

    

    for epoch in range(num_epochs):

        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples

        for i, (X, y) in enumerate(train_iter):

            timer.start()

            net.train()

            optimizer.zero_grad()

            X, y = X.to(device), y.to(device)

            y_hat = net(X)

            l = loss(y_hat, y)

            l.backward()

            optimizer.step()

            with torch.no_grad():

                metric.add(l*X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])

            timer.stop()

            train_loss, train_acc = metric[0]/metric[2], metric[1]/metric[2]

            if (i+1) % 5 == 0:

                animator.add(epoch + i/len(train_iter),

                             (train_loss, train_acc, None))

                #print('%.1f examples/sec on %s' % (metric[2]/timer.sum(), device))

                #print(train_loss)

                #print(train_acc)

    

        

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)

        

        animator.add(epoch+1, (None, None, test_acc))

        print('loss %.3f, train acc %.3f, test acc %.3f' % (

        train_loss, train_acc, test_acc))

        print('%.1f examples/sec on %s' % (

        metric[2]*num_epochs/timer.sum(), device))
lr, num_epochs = 0.03, 10

train_ch6(net, train_iter, val_iter, num_epochs, lr)
torch.save(net.state_dict(), 'saved_para3.pth')