import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision

from torchvision import transforms, models

from torch.utils.data import Dataset, DataLoader

import torchvision.datasets as datasets

import numpy as np

import random

import PIL

from PIL import Image

from pathlib import Path

import matplotlib.pyplot as plt

import tqdm

from tqdm import tqdm_notebook

import pandas as pd

import glob

import os

%matplotlib inline
Path.ls = lambda x: list(x.iterdir())
!ls ../input/
PATH_TRAIN = '../input/train'
files = []

for fn in Path(PATH_TRAIN).ls():

    files.extend(fn.ls())
len(files)
files[0].parent.name
df = pd.DataFrame()

for idx, file in enumerate(files):

    if idx%1000==0: print(f'{idx+1}/{len(files)}')

    face_id = files[idx].name.split('.')[0]

    face_label = files[idx].parent.name

    df = df.append({'id': face_id, 'name': face_label}, ignore_index=True)
df.head()
df = df.sort_values(by=['name', 'id']).reset_index(drop=True)
df.head()
df['class'] = pd.factorize(df['name'])[0]
class MyDataset(Dataset):

    def __init__(self, path, df, num_triplets, tfms=None):

        self.path = path

        self.df = df

        self.num_triplets = num_triplets

        self.tfms = tfms

        self.training_triplets = self.generate_triplets(self.df, self.num_triplets)

    @staticmethod

    def generate_triplets(df, num_triplets):

        def make_dict_for_face_class(df):

            face_classes = dict()

            for idx, label in enumerate(df['class']):

                if label not in face_classes: face_classes[label] = []

                face_classes[label].append(df.iloc[idx, 0])

            return face_classes

        triplets = []

        classes = df['class'].unique()

        face_classes = make_dict_for_face_class(df)

        for _ in range(num_triplets):

            pos_class = np.random.choice(classes)

            neg_class = np.random.choice(classes)

            while len(face_classes[pos_class])<2: pos_class = np.random.choice(classes)

            while pos_class==neg_class: neg_class = np.random.choice(classes)

            pos_name = df.loc[df['class']==pos_class, 'name'].values[0]

            neg_name = df.loc[df['class']==neg_class, 'name'].values[0]

            if len(face_classes[pos_class])==2:

                ianc, ipos = np.random.choice(2, size=2, replace=False)

            else:

                ianc = np.random.randint(0, len(face_classes[pos_class]))

                ipos = np.random.randint(0, len(face_classes[pos_class]))

                while ianc==ipos:

                    ipos = np.random.randint(0, len(face_classes[pos_class]))

            ineg = np.random.randint(0, len(face_classes[neg_class]))

            triplets.append([face_classes[pos_class][ianc], face_classes[pos_class][ipos], 

                             face_classes[neg_class][ineg], pos_class, neg_class, pos_name, neg_name])

        return triplets

    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]

        anc_img = self.path/pos_name/(str(anc_id)+'.jpg')

        pos_img = self.path/pos_name/(str(pos_id)+'.jpg')

        neg_img = self.path/neg_name/(str(neg_id)+'.jpg')

        anc_img = PIL.Image.open(anc_img)

        pos_img = PIL.Image.open(pos_img)

        neg_img = PIL.Image.open(neg_img)

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))

        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {

            'anc_img': anc_img,

            'pos_img': pos_img,

            'neg_img': neg_img,

            'pos_class': pos_class,

            'neg_class': neg_class

        }

        if self.tfms: 

            sample['anc_img'] = self.tfms(sample['anc_img'])

            sample['pos_img'] = self.tfms(sample['pos_img'])

            sample['neg_img'] = self.tfms(sample['neg_img'])

        return sample

    def __len__(self): return len(self.training_triplets)
num_train_triplets = 5000

batch_size = 32

num_workers = 0
train_tfms = transforms.Compose([

    transforms.Resize(256),

    transforms.RandomHorizontalFlip(),

    transforms.RandomCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

train_ds = MyDataset(Path('../input/train'), df, num_train_triplets, tfms=train_tfms)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

train_data_size = len(train_ds)

train_data_size
class Net(nn.Module):

    def __init__(self, emb_size, num_classes, pretrained=False):

        super(Net, self).__init__()

        self.model = models.resnet34(pretrained)

        self.emb_size = emb_size 

        self.lin1 = nn.Linear(1000, emb_size)

        self.lin2 = nn.Linear(emb_size, num_classes)

    def l2_norm(self, input):

        input_size = input.size()

        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)

        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):

        x = self.model(x)

        x = self.lin1(x)

        self.features = self.l2_norm(x)

        alpha = 10

        self.features*=10

        return self.features

    def forward_classifier(self, x):

        features = self.forward(x)

        res = self.lin2(features)

        return res
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(128, 5000, pretrained=True).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
from torch.nn.modules.distance import PairwiseDistance

l2_dist = PairwiseDistance(2)
margin = 0.5
class TripletLoss(nn.Module):

    def __init__(self, margin):

        super(TripletLoss, self).__init__()

        self.margin = margin

        self.pdist = PairwiseDistance(2)

    def forward(self, anc, pos, neg):

        pos_dist = self.pdist(anc, pos)

        neg_dist = self.pdist(anc, neg)

        hinge_dist = torch.clamp(self.margin+pos_dist-neg_dist, min=0.0)

        loss = torch.mean(hinge_dist)

        return loss
for epoch in range(10):

    triplet_loss_sum = 0.0

    scheduler.step()

    model.train()

    for batch_idx, batch_sample in tqdm_notebook(enumerate(train_dl)):

        anc_img = batch_sample['anc_img'].to(device)

        pos_img = batch_sample['pos_img'].to(device)

        neg_img = batch_sample['neg_img'].to(device)

        pos_cls = batch_sample['pos_class'].to(device)

        neg_cls = batch_sample['neg_class'].to(device)

        anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)

        pos_dist = l2_dist(anc_embed, pos_embed)

        neg_dist = l2_dist(anc_embed, neg_embed)

        all = (neg_dist - pos_dist < margin).cpu().numpy().flatten()

        hard_triplets = np.where(all==1)

        if len(hard_triplets)==0: continue

        anc_hard_embed = anc_embed[hard_triplets].to(device)

        pos_hard_embed = pos_embed[hard_triplets].to(device)

        neg_hard_embed = neg_embed[hard_triplets].to(device)

        triplet_loss = TripletLoss(margin)(anc_hard_embed, pos_hard_embed, neg_hard_embed).to(device)

        optimizer.zero_grad()

        triplet_loss.backward()

        optimizer.step()

        triplet_loss_sum+=triplet_loss.item()

    avg_loss = triplet_loss_sum/len(train_dl.dataset)

    print('Epoch', epoch)

    print('Loss: ', avg_loss)
test_df = pd.read_csv('../input/test_final.csv')
test_df.head()
test_tfms = transforms.Compose([

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])
class TestDataset(Dataset):

    def __init__(self, path, test_df, test_tfms=None):

        self.path = path

        self.test_tfms = test_tfms

        self.images1 = test_df['image1'].tolist()

        self.images2 = test_df['image2'].tolist()

    def __getitem__(self, idx):

        img1 = PIL.Image.open(self.path/self.images1[idx])

        img2 = PIL.Image.open(self.path/self.images2[idx])

        if test_tfms is not None:

            img1 = self.test_tfms(img1)

            img2 = self.test_tfms(img2)

        return img1, img2

    def __len__(self): return len(self.images1)
test_ds = TestDataset(Path('../input/test'), test_df, test_tfms=test_tfms)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
preds = []
model.eval()
sample_test = next(iter(test_dl))
out1, out2 = model(sample_test[0].cuda()), model(sample_test[1].cuda())
out1.shape
def pairwise_distances(x, y=None):

    '''

    Input: x is a Nxd matrix

           y is an optional Mxd matirx

    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]

            if y is not given then use 'y=x'.

    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2

    '''

    x_norm = (x**2).sum(1).view(-1, 1)

    if y is not None:

        y_norm = (y**2).sum(1).view(1, -1)

    else:

        y = x

        y_norm = x_norm.view(1, -1)



    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))

    return dist
np.linalg.norm(out1.cpu().detach().numpy()-out2.cpu().detach().numpy(), axis=1)
preds = []
for i, data in tqdm_notebook(enumerate(test_dl)):

    img1, img2 = data

    img1 = img1.cuda()

    img2 = img2.cuda()

    output1, output2 = model(img1), model(img2)

    dist = np.linalg.norm(output1.cpu().detach().numpy()-output2.cpu().detach().numpy(), axis=1)

    preds.extend(dist.tolist())
preds
x = [(i, p) for i, p in enumerate(preds) if p>1.09584 and p<1.10218]
x
PIL.Image.open('../input/test/' + imgs2[270])
len([p for p in preds if p>1.10])
sub_df = pd.read_csv('../input/test_final.csv')
imgs1 = test_df['image1'].tolist()

imgs2 = test_df['image2'].tolist()
pred = [0 if p>1.25 else 1 for p in preds]
sub_df['target'] = pred
sub_df.to_csv('sub.csv', index=False)
from IPython.display import FileLink
FileLink('sub.csv')