# Import python library dependencies

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from matplotlib import pyplot as plt

%matplotlib inline



import os

from PIL import Image



import torch

import torchvision

from torchvision import models

import torch.nn as nn

import torch.optim

import torch.utils.data

import torchvision.transforms as transforms

import torchvision.datasets as datasets

from torchvision.utils import make_grid
# Define the dataloader

train_transform = transforms.Compose([

        transforms.Resize(256),

        transforms.RandomCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.RandomVerticalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

dataset_train = datasets.ImageFolder('/kaggle/input/cuhksz-facecomp/trains/trains/', train_transform)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True, pin_memory=True, num_workers=8)
batch = next(iter(train_loader))

plt.imshow(make_grid(batch[0]).cpu().numpy().transpose((1,2,0)))
# Use pre-trained resnet18 as our model

num_class = len(train_loader.dataset.classes)



model = torchvision.models.resnet50(pretrained=True)

print(model)

model.fc = torch.nn.Linear(2048, num_class)

model = model.cuda()
# Define the metric and optimizer

criterion = nn.CrossEntropyLoss()

optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()) , lr=1e-4)
def eval_metric(lbl_pred, lbl_true):

    accu = []

    for lt, lp in zip(lbl_true, lbl_pred):

        accu.append(np.mean(lt == lp))

    return np.mean(accu)



for epoch in range(10):

    for batch_idx, (data, target) in enumerate(train_loader):

        optim.zero_grad()

        data, target = data.cuda(), target.cuda()

        score = model(data)

        loss = criterion(score, target)

        loss.backward()

        optim.step()

        

        lbl_pred = score.data.max(1)[1].cpu().numpy()

        lbl_pred = lbl_pred.squeeze()

        lbl_true = target.data.cpu()

        lbl_true = np.squeeze(lbl_true.numpy())

        train_accu = eval_metric([lbl_pred], [lbl_true])

        

        if batch_idx % 20 == 0:

            print(epoch, batch_idx, loss.item(), train_accu)
class Identity(nn.Module):

    def __init__(self):

        super().__init__()



    def forward(self, x):

        return x

model.fc = Identity()
def l2_dist(x1, x2):

    assert x1.size() == x2.size()

    eps = 1e-4 / x1.size(1)

    diff = torch.abs(x1 - x2)

    out = torch.pow(diff, 2).sum(dim=1)

    return torch.pow(out + eps, 1. / 2)
submits = []

peoples = os.listdir('/kaggle/input/cuhksz-facecomp/tests/tests/')

images_key = []

images_dict = {}



model.eval()

for p in peoples:

    path = os.path.join('/kaggle/input/cuhksz-facecomp/tests/tests/', p)

    img1, img2 = os.listdir(path)

    img1 = os.path.join('/kaggle/input/cuhksz-facecomp/tests/tests/', p, img1)

    img2 = os.path.join('/kaggle/input/cuhksz-facecomp/tests/tests/', p, img2)

    

    img1, img2 = Image.open(img1), Image.open(img2)

    img1_th, img2_th = transforms.ToTensor()(img1), transforms.ToTensor()(img2)

    img1_th, img2_th = img1_th.unsqueeze(0).cuda(), img2_th.unsqueeze(0).cuda()

    with torch.no_grad():

        img1_th_emb, img2_th_emb = model(img1_th), model(img2_th)

    distance = torch.nn.functional.mse_loss(img1_th_emb, img2_th_emb).item()

    distance /= 0.4

    submits.append([p, 1 if distance > 1 else distance])
submits
import csv

with open('submission.csv', 'w') as csvfile:  

    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(['id', 'similar'])  

    csvwriter.writerows(submits)