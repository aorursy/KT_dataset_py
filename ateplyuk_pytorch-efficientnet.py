import numpy as np

import pandas as pd

import os

import matplotlib.image as mpimg



import torch

import torch.nn as nn

import torch.optim as optim 



import torchvision

from torch.utils.data import DataLoader, Dataset

import torch.utils.data as utils

from torchvision import transforms



import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
data_dir = '../input'

train_dir = data_dir + '/train/train/'

test_dir = data_dir + '/test/test/'
labels = pd.read_csv("../input/train.csv")

labels.head()
class ImageData(Dataset):

    def __init__(self, df, data_dir, transform):

        super().__init__()

        self.df = df

        self.data_dir = data_dir

        self.transform = transform



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):       

        img_name = self.df.id[index]

        label = self.df.has_cactus[index]

        

        img_path = os.path.join(self.data_dir, img_name)

        image = mpimg.imread(img_path)

        image = self.transform(image)

        return image, label
data_transf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

train_data = ImageData(df = labels, data_dir = train_dir, transform = data_transf)

train_loader = DataLoader(dataset = train_data, batch_size = 64)
!pip install efficientnet_pytorch
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_name('efficientnet-b1')
# Unfreeze model weights

for param in model.parameters():

    param.requires_grad = True
num_ftrs = model._fc.in_features

model._fc = nn.Linear(num_ftrs, 1)
model = model.to('cuda')
optimizer = optim.Adam(model.parameters())

loss_func = nn.BCELoss()
%%time

# Train model

loss_log = []



for epoch in range(5):    

    model.train()    

    for ii, (data, target) in enumerate(train_loader):

        data, target = data.cuda(), target.cuda()

        target = target.float()                



        optimizer.zero_grad()

        output = model(data)                

    

        m = nn.Sigmoid()

        loss = loss_func(m(output), target)

        loss.backward()



        optimizer.step()  

        

        if ii % 1000 == 0:

            loss_log.append(loss.item())

       

    print('Epoch: {} - Loss: {:.6f}'.format(epoch + 1, loss.item()))
# plt.figure(figsize=(10,8))

# plt.plot(loss_log)
submit = pd.read_csv('../input/sample_submission.csv')

test_data = ImageData(df = submit, data_dir = test_dir, transform = data_transf)

test_loader = DataLoader(dataset = test_data, shuffle=False)
%%time

predict = []

model.eval()

for i, (data, _) in enumerate(test_loader):

    data = data.cuda()

    output = model(data)    



    pred = torch.sigmoid(output)

    predicted_vals = pred > 0.5

    predict.append(int(predicted_vals))

    

submit['has_cactus'] = predict

submit.to_csv('submission.csv', index=False)
submit.head()