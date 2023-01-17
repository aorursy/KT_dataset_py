# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import sys



print(os.listdir("../input"))

print(os.listdir("../input/seg_train/seg_train"))

# Any results you write to the current directory are saved as output.
import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision

from torchvision import transforms

from torch.utils.data import DataLoader

from PIL import Image

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix, accuracy_score

from IPython.display import display

import cv2

import matplotlib.image as mtim

import matplotlib.pyplot as plt

import tqdm

from tqdm import tqdm_notebook

import random
seed = 50

np.random.seed(seed)

torch.manual_seed(seed)
train_path = '../input/seg_train/seg_train/'

test_path = "../input/seg_test/seg_test/"

pred_path = "../input/seg_pred/seg_pred/"

classes = dict(enumerate(os.listdir(test_path)))

print(classes)
class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, imgs_path, root_path, transforms=None):

        super().__init__()

        if root_path[-1] != '/':

            root_path += '/'

        self.root_path = root_path

        self.imgs_path = imgs_path

        self.transforms = transforms

        

    def __len__(self):

        return len(self.imgs_path)

    

    def __getitem__(self, idx):

        img = self.get_image(idx)

        

        if self.transforms:

            img = self.transforms(img)

        return (img, self.imgs_path[idx][1])

    

    def get_image(self, idx):

        img = Image.open(self.root_path + self.imgs_path[idx][0])

        return img
def get_ImageDataset(path, transform = True):

    path += '/' if path[-1] != '/' else ''

    classes = os.listdir(path)

    imgs_path = []

    counter = 0

    for cl in classes:

        for i in os.listdir(path + cl):

            imgs_path.append((cl + '/' + i, counter))

        counter += 1

        

    _mean = [.485, .456, .406]

    _std = [.229, .224, .225]

    trans = transforms.Compose([

        transforms.Resize((150, 150)),

        transforms.RandomHorizontalFlip(),

        transforms.RandomCrop((75,75)),

        transforms.ToTensor(),

        transforms.Normalize(_mean, _std),

    ])

    if transform:

        return ImageDataset(imgs_path, path, trans)

    return ImageDataset(imgs_path, path)
def get_test_valid_ImageDataset(path, valid_size, transform = True):

    path += '/' if path[-1] != '/' else ''

    classes = os.listdir(path)

    imgs_path = []

    counter = 0

    for cl in classes:

        for i in os.listdir(path + cl):

            imgs_path.append((cl + '/' + i, counter))

        counter += 1

        

    _mean = [.485, .456, .406]

    _std = [.229, .224, .225]

    trans = transforms.Compose([

        transforms.Resize((150, 150)),

        transforms.RandomHorizontalFlip(),

        transforms.RandomCrop((75,75)),

        transforms.ToTensor(),

        transforms.Normalize(_mean, _std),

    ])

    valid_path = []

    for i in range(int(valid_size*len(imgs_path))):

        idx = random.randint(0, len(imgs_path)-1)

        valid_path.append(imgs_path.pop(idx))

    if transform:

        return ImageDataset(imgs_path, path, trans), ImageDataset(valid_path,path, trans)

    return ImageDataset(imgs_path, path), ImageDataset(valid_path, path)
test_ds, val_ds = get_test_valid_ImageDataset(test_path, 0.4)

train_ds = get_ImageDataset(train_path)



train_dl = DataLoader(train_ds, batch_size=30, shuffle=True)

val_dl = DataLoader(val_ds, batch_size=30, shuffle=True)

test_dl = DataLoader(test_ds, batch_size=30, shuffle=False)
ys = []

for _, yy in val_dl:

    ys.extend(yy.tolist())

print(classification_report(ys, ys))
print(len(val_dl))

print(len(test_dl))

print(len(train_dl))
class Net(nn.Module):

    def __init__(self):

        super().__init__()

        # 3* 150

        self.conv1 = nn.Conv2d(3, 25, 11)

        # 60* 140

        self.pool1 = nn.MaxPool2d(4, 4)

        # 60* 35

        self.conv3 = nn.Conv2d(25, 80, 6)

        # 100* 30

        self.conv4 = nn.Conv2d(80, 200, 6)

        # 200* 25

        self.pool2 = nn.AvgPool2d(25, 25)

        self.l1 = nn.Linear(200, 6)

        

        

    def forward(self, x):

        x = self.conv1(x)

        x = F.relu(self.pool1(x))    

        x = self.conv4(self.conv3(x))

        x = F.relu(self.pool2(x))

        x = x.view(-1, 200)

        x = self.l1(x)

        return x

    

    

    def predict(self, x):

        return F.log_softmax(self.forward(x), dim=1)

    

        

model = Net().cuda() if torch.cuda.is_available() else Net()

print(model)
def fit(model, train_dl, valid_dl, lr=3e-4, epoches=5):

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_state = None

        best_acc = 0

        for epoche in range(epoches):

            model.train()

            print("Training")

            ep_loss = 0

            for num, (xx, yy) in tqdm_notebook(enumerate(train_dl, 0), total=len(train_dl)):

                if(torch.cuda.is_available()):

                    xx, yy = xx.cuda(), yy.cuda()

                optimizer.zero_grad()

                y_ = model(xx)

                loss = criterion(y_, yy)

                loss.backward()

                ep_loss+=loss.item()

                optimizer.step()

            print("Loss: {}".format(ep_loss/len(train_dl)))

        

            model.eval()

            y_pred = []

            y_true = []

            print("Validation")

            with torch.no_grad():

                for num, (xx, yy) in tqdm_notebook(enumerate(valid_dl), total=len(valid_dl)):

                    y_true.extend(yy.tolist())

                    if(torch.cuda.is_available()):

                        xx, yy = xx.cuda(), yy.cuda()

                    y_ = model.predict(xx).argmax(dim=1)

                    y_pred.extend(y_.tolist())

            print("Epoche {}".format(epoche))

            #print(classification_report(y_true, y_pred))

            print(confusion_matrix(y_true, y_pred))

            ep_acc = accuracy_score(y_true, y_pred)

            print(ep_acc)

            if best_acc < ep_acc:

                best_acc = ep_acc

                best_state = model.state_dict()

            elif best_acc > ep_acc:

                model.load_state_dict(best_state)

                print("Ep_acc({:.2}) < Prev_acc({:.2})".format(ep_acc, best_acc))

                print("Model state was reloaded")
fit(model, train_dl, val_dl, epoches=5)
model.eval()

y_true = []

y_pred = []

for num ,(xx, yy) in tqdm_notebook(enumerate(test_dl), total=(len(test_dl))):

    if torch.cuda.is_available():

        xx, yy = xx.cuda(), yy.cuda()

    out = model(xx).argmax(dim=1)

    y_true.extend(yy.tolist())

    y_pred.extend(out.tolist())

print(classification_report(y_true, y_pred))

print(confusion_matrix(y_true, y_pred))

    
with torch.no_grad():

    params = list(model.parameters())

    ps = params[2][1].to("cpu").numpy()

    print(len(ps))

    img = mtim.imread(pred_path + os.listdir(pred_path)[9])

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    fig=plt.figure(figsize=(10, 10))

    k = 1

    j = 1

    for i in range(0,len(ps)+1):

        fig.add_subplot(k, len(ps)+1, j)

        if(j%5 == 0):

            j = 1

        if(i%5 == 0):

            k += 1

        j += 1

        if i == 0:

            plt.imshow(img)        

        else:

            f_img = cv2.filter2D(img_gray, -1, ps[i-1])

            plt.imshow(f_img, cmap='gray')

    plt.show()
import random

_mean = [.485, .456, .406]

_std = [.229, .224, .225]

trans = transforms.Compose([

    transforms.Resize((150, 150)),

    transforms.ToTensor(),

    transforms.Normalize(_mean, _std)])

imgs_path = []

batch_size = 5

imgs = random.choices(os.listdir(pred_path), k=batch_size)

print(imgs)

for i in range(batch_size):

    imgs_path.append((imgs[i],0))

pred_ds = ImageDataset(imgs_path, pred_path, trans)

pred_dl = DataLoader(pred_ds)
model.eval()

counter = 0

for xx, yy in pred_dl:

    if torch.cuda.is_available():

        xx = xx.cuda()

    out = model(xx).argmax(dim=1).item()

    print("This is a {}".format(classes[out]))

    display(pred_ds.get_image(counter))

    counter += 1
import gc

gc.collect()