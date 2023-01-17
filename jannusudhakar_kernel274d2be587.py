import numpy as np # linear algebra

import torch

from torch import nn

from torch.nn import functional as F

import os

from matplotlib import pyplot as plt

from PIL import Image
im_size = 320

train_normal = []

train_pneumonia = []

print("starting...")

for f in os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL'):

    filename = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL',f)

    image = Image.open(filename)

    image = np.asarray(image.resize([im_size,im_size]))

    if(len(image.shape) == 3):

        image = image.mean(axis = 2)

    train_normal.append(np.asarray(image))

for f in os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA'):

    filename = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA',f)

    image = Image.open(filename)

    image = np.asarray(image.resize([im_size,im_size]))

    if(len(image.shape) == 3):

        image = image.mean(axis = 2)

    train_pneumonia.append(image)

print("len(train_normal):", len(train_normal))

print("len(train_pneumonia):", len(train_pneumonia))



test_normal = []

test_pneumonia = []

for f in os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL'):

    filename = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL',f)

    image = Image.open(filename)

    image = np.asarray(image.resize([im_size,im_size]))

    if(len(image.shape) == 3):

        image = image.mean(axis = 2)

    test_normal.append(image)

for f in os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA'):

    filename = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA',f)

    image = Image.open(filename)

    image = np.asarray(image.resize([im_size,im_size]))

    if(len(image.shape) == 3):

        image = image.mean(axis = 2)

    test_pneumonia.append(image)

print("len(test_normal):", len(test_normal))

print("len(test_pneumonia):", len(test_pneumonia))

val_normal = []

val_pneumonia = []

for f in os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL'):

    filename = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL',f)

    image = Image.open(filename)

    image = np.asarray(image.resize([im_size,im_size]))

    if(len(image.shape) == 3):

        image = image.mean(axis = 2)

    val_normal.append(image)

for f in os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA'):

    filename = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA',f)

    image = Image.open(filename)

    image = np.asarray(image.resize([im_size,im_size]))

    if(len(image.shape) == 3):

        image = image.mean(axis = 2)

    val_pneumonia.append(image)

print("len(val_normal):", len(val_normal))

print("len(val_pneumonia):", len(val_pneumonia))
class model(torch.nn.Module):

    def __init__(self):

        super(model,self).__init__()

        self.pool = torch.nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0)

        self.dropout1 = nn.Dropout(0)

        self.conv1 = nn.Conv2d(1,15,3)

        self.conv2 = nn.Conv2d(15,15,3)

        self.conv3 = nn.Conv2d(15,15,3)

        self.conv4 = nn.Conv2d(15,15,3)

        self.conv5 = nn.Conv2d(15,15,3)

        self.conv6 = nn.Conv2d(15,4,3)

        

        self.linear1 = nn.Linear(256,256)

        self.linear2 = nn.Linear(256,256)

        self.linear3 = nn.Linear(256,256)

        self.linear4 = nn.Linear(256,256)

        self.linear5 = nn.Linear(256,1)

        

    def forward(self,x):

        #print("1:",x.norm().item())

        x = self.dropout(F.relu(self.conv1(x)))

        #print("2:",x.norm().item())

        x = self.dropout(F.relu(self.pool(self.conv2(x))))

        #print("3:",x.norm().item())

        x = self.dropout(F.relu(self.pool(self.conv3(x))))

        #print("4:",x.norm().item())

        x = self.dropout(F.relu(self.pool(self.conv4(x))))

        #print("5:",x.norm().item())

        x = self.dropout(F.relu(self.pool(self.conv5(x))))

        #print("6:",x.norm().item())

        x = self.dropout(F.relu(self.pool(self.conv6(x))))

        #print("7:",x.norm().item())

        x = x.view(-1,256)

        #print("8:",x.norm().item())

        x = self.dropout1(F.relu(self.linear1(x)))

        #print("9:",x.norm().item())

        x = self.dropout1(F.relu(self.linear2(x)))

        #print("10:",x.norm().item())

        x = self.dropout1(F.relu(self.linear3(x)))

        #print("11:",x.norm().item())

        x = self.dropout1(F.relu(self.linear4(x)))

        #print("12:",x.norm().item())

        x = self.linear5(x)

        #print("13:",x.norm().item())

        return x.view(-1).sigmoid()
torch.manual_seed(0)

net = model()

net = net.cuda()

criterion =  nn.BCELoss()

optimizer = torch.optim.Adam(net.parameters(),lr = 0.00005)
train_in = train_normal + train_pneumonia

train_out = [0]*len(train_normal) + [1]*len(train_pneumonia)

test_in = test_normal + test_pneumonia

test_out = [0]*len(test_normal) + [1]*len(test_pneumonia)

rand_permutation = np.random.permutation(len(train_in))
#training

batch_size = 8

for epoch in range(1):

    true_positives = 0

    false_positives = 0

    true_negatives = 0

    false_negatives = 0

    net.train()

    for i in range(len(train_out)//batch_size):

        x = []

        y = []

        for j in range(batch_size):

            x.append(train_in[rand_permutation[i*batch_size + j]])

            y.append(train_out[rand_permutation[i*batch_size + j]])

        x = torch.tensor(np.array(x)).float().cuda().view(batch_size,1,im_size,im_size)

        y = torch.tensor(np.array(y)).float().cuda()

        h = net(x)

        loss = criterion(h,y)

        #print(loss.item())

        loss.backward()

        #for p in net.parameters():

        #    print(p.grad.norm().item())

        optimizer.step()

        optimizer.zero_grad()

        h = (h.detach().cpu().numpy() > 0.5).astype(int)

        y = y.cpu().numpy().astype(int)

        true_positives += np.sum((h==y)*h)

        true_negatives += np.sum((h==y)*(1-h))

        false_positives += np.sum((h!=y)*h)

        false_negatives += np.sum((h!=y)*(1-h))

    precision = true_positives/(true_positives + false_positives)

    recall = true_positives/(true_positives + false_negatives)

    f1_score = 2/((1/precision) + (1/recall))

    print("epoch %d train:- precision: %f, recall: %f, f1 score: %f"%(epoch,precision,recall,f1_score))

    print("      train:- tp:%d,fp:%d,tn:%d,fn:%d"%(true_positives,false_positives,true_negatives,false_negatives))

    true_positives = 0

    false_positives = 0

    true_negatives = 0

    false_negatives = 0

    net.eval()

    for i in range(len(test_out)//batch_size):

        x = []

        y = []

        for j in range(batch_size):

            x.append(test_in[i*batch_size + j])

            y.append(test_out[i*batch_size + j])

        x = torch.tensor(np.array(x)).float().cuda().view(batch_size,1,im_size,im_size)

        y = np.array(y)

        h = net(x)

        h = (h.detach().cpu().numpy() > 0.5).astype(int)

        #print(h,y)

        true_positives += np.sum((h==y)*h)

        true_negatives += np.sum((h==y)*(1-h))

        false_positives += np.sum((h!=y)*h)

        false_negatives += np.sum((h!=y)*(1-h))

    precision = true_positives/(true_positives + false_positives)

    recall = true_positives/(true_positives + false_negatives)

    f1_score = 2/((1/precision) + (1/recall))

    print("      test:- precision: %f, recall: %f, f1 score: %f"%(precision,recall,f1_score))

    print("      test:- tp:%d,fp:%d,tn:%d,fn:%d"%(true_positives,false_positives,true_negatives,false_negatives))
