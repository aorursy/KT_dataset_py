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

import cv2

import torch

import torch.nn as nn

from PIL import Image

import matplotlib.pyplot as plt

import seaborn as sns

import torchvision

from torchvision import transforms

from tqdm import tqdm_notebook as tqdm

from torch.utils.data import Dataset,DataLoader

import warnings

warnings.filterwarnings("ignore")
path = os.path.join("/kaggle/input/coic2")

os.listdir(path)
train_path = os.path.join(path,'flowers')

#classes = {label: idx for idx, label in enumerate(os.listdir(os.path.join(train_path)))}

classes = {'daisy':0 ,'dandelion':1, 'rose':2,'sunflower':3,'tulip':4}

classes
def make_samples():

    images = []

    

    for class_name in sorted(classes.keys()):



        dir = os.path.join(train_path,class_name)

        

        for root, _, fnames in sorted(os.walk(dir,followlinks=True)):

            for fname in sorted(fnames):

                if '.jpg' in fname:

                    path = os.path.join(root,fname)

                    item = (path, classes[class_name])

                    images.append(item)

    images = pd.DataFrame(images).rename(columns={0:'fname',1:'label'})

    return images

samples = make_samples()
ax = sns.countplot(samples.label.map({idx:i for idx,i in enumerate(classes)}))
transform = transforms.Compose([

    transforms.RandomRotation(10,resample=False, expand=False, center=None),

    transforms.CenterCrop(224),

    transforms.RandomVerticalFlip(p=0.2),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ToTensor(),

    transforms.Normalize((0, 0, 0), (1, 1, 1)),

])
class FlowerDataset(Dataset):

    

    def __init__(self,df,transforms = None):

        

        self.df = df

        self.img = self.df.fname.values

        if 'label' not in self.df.columns:

            self.label = None

        else:

            self.label = self.df.label.values

        self.transforms = transforms

        

    def __len__(self):

        

        return len(self.img)

    

    def __getitem__(self,idx):

        

        img = cv2.imread(self.img[idx])

        # Change BGR to RGB

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img,(224,224))

        if self.label is not None:

            target = self.label[idx]

            if self.transforms:

                img = Image.fromarray(img)

                img = self.transforms(img)

                #permute dimensions compatible with before augmentation and pytorch format

                return torch.tensor(img).permute(0,2,1),target

            else:

                # rearrange dimensions compatible with pytorch format

                return torch.tensor(img).permute(2,1,0),target

        else:

            return torch.tensor(img).permute(2,1,0)
train_images = FlowerDataset(samples)

train_dataloader = torch.utils.data.DataLoader(train_images, batch_size=6, shuffle=True)

a = next(iter(train_dataloader))

fig,ax = plt.subplots(2,3, figsize=(25,25))

for i in range(6):

    j = i//3

    k = i%3

    ax[j,k].imshow(torch.transpose(a[0][i],0,2))

    ax[j,k].set_title(list(classes.keys())[a[1][i]])

plt.show()
train_images = FlowerDataset(samples,transforms=transform)

train_dataloader = torch.utils.data.DataLoader(train_images, batch_size=6, shuffle=True)

a = next(iter(train_dataloader))

fig,ax = plt.subplots(2,3, figsize=(25,25))

for i in range(6):

    j = i//3

    k = i%3

    ax[j,k].imshow(torch.transpose(a[0][i],0,2))

    ax[j,k].set_title(list(classes.keys())[a[1][i]])

plt.show()
from torchvision import transforms, models
class FlowerNet(nn.Module):

    def __init__(self):

        super(FlowerNet, self).__init__()

        self.resnet = models.resnet152(pretrained=True, progress=False)

        self.fc1 = nn.Linear(1000,500)

        self.fc2 = nn.Linear(500,128)

        self.fc3 = nn.Linear(128, 5)

        for param in self.resnet.parameters():

            param.requires_grad = False

    def forward(self,x):

        x = self.resnet(x)

        # Adding classifier layers

        x = self.fc1(x)

        x = self.fc2(x)

        x = self.fc3(x)

        return x
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FlowerNet().to(device)
!pip install torchsummary

from torchsummary import summary
summary(model,(3,224,224)) #input = (Channel,Height,Width)
train_losses = []

train_accs = []

val_losses = []

val_accs = []

def Train(model):

    model.train()

    epochs=10

    for epoch in range(epochs):

        print('epochs : {}/{}'.format(epoch+1,epochs))

        running_loss = 0.

        running_acc = 0.

        for idx, (inputs,labels) in tqdm(enumerate(train_loader),total=len(train_loader)):

            optimizer.zero_grad()

            inputs = inputs.to(device)

            labels = labels.to(device)

            preds = model(inputs.float())

            loss_val = loss(preds, labels)

            pred_classes = preds.argmax(dim=1)

            loss_val.backward()

            optimizer.step()

            running_loss += loss_val

            running_acc += (pred_classes == labels.data).float().mean()

        

        train_loss = running_loss/len(train_loader)

        train_acc = running_acc/len(train_loader)

        print('train_loss : {:.4f}, train_acc : {:.2f}%'.format(train_loss,train_acc))

        train_losses.append(train_loss)

        train_accs.append(train_acc)

        Valid(model)

    return model

def Valid(model):

    #model.eval()

    running_loss = 0.

    running_acc = 0.

    with torch.no_grad():

        

        for idx, (inputs,labels) in enumerate(valid_loader):

            

            inputs = inputs.to(device)

            labels = labels.to(device)

            

            preds = model(inputs.float())

            loss_val = loss(preds,labels)

            pred_classes = preds.argmax(dim=1)

            

            running_loss += loss_val

            running_acc += (pred_classes == labels.data).float().mean()

            

        val_loss = running_loss/len(valid_loader)

        val_acc = running_acc/len(valid_loader)

        print('val_loss : {:.2f}, val_acc : {:.2f}%'.format(val_loss,val_acc))

        val_losses.append(val_loss)

        val_accs.append(val_acc)

def Test(model):

    prediction = []

    test_images = FlowerDataset(pd.DataFrame(test_samples))

    test_loader = torch.utils.data.DataLoader(test_images, batch_size=32, shuffle=False)

    model.eval()



    for idx, (inputs) in enumerate(test_loader):

            inputs = inputs.to(device)

            preds = model(inputs.float())

            pred_classes = preds.argmax(dim=1)

            prediction.append(pred_classes.cpu().detach().numpy())

    prediction = np.hstack(prediction)

    return prediction

            
'''

from sklearn.model_selection import train_test_split

train_set, valid_set, _, _ = train_test_split(samples, samples.label,

                                                    stratify=samples.label, 

                                                    test_size=0.05)

'''
def make_test_samples():

    images = []

    

    path = '/kaggle/input/coic2'

    dir = os.path.join(path,'unlabel-flowers')

        

    for root, _, fnames in os.walk(dir,followlinks=True):

        for fname in fnames:

            if '.jpg' in fname:

                path = os.path.join(root,fname)

                item = (path)

                images.append(item)

    images = pd.DataFrame(images).rename(columns= {0:'fname'})

    return images

test_samples = make_test_samples()
from sklearn.model_selection import StratifiedKFold

folds  = 10

skf = StratifiedKFold(n_splits=folds, random_state=7, shuffle=True)
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
for idx, (train_idx,valid_idx) in enumerate(skf.split(samples,samples.label)):

    train_images = FlowerDataset(samples.iloc[train_idx],transforms=transform)

    train_loader = torch.utils.data.DataLoader(train_images, batch_size=32, shuffle=True)

    valid_images = FlowerDataset(samples.iloc[valid_idx])

    valid_loader = torch.utils.data.DataLoader(valid_images, batch_size=32, shuffle=True)

    model = Train(model)

prediction = Test(model)
prediction = Test(model)
fig, axs = plt.subplots(2, 2, figsize=(15, 5))

axs[0,0].plot(train_accs)

axs[0,0].set_title('Train Accuracy')

axs[0,1].plot(train_losses)

axs[0,1].set_title('Train Loss')

axs[1,0].plot(val_accs)

axs[1,0].set_title('Val Accuracy')

axs[1,1].plot(val_losses)

axs[1,1].set_title('Val Loss')

fig.tight_layout()
test_samples['label'] = prediction

test_images = FlowerDataset(test_samples)

test_loader = torch.utils.data.DataLoader(test_images, batch_size=15, shuffle=True)
a = next(iter(test_loader))

fig,ax = plt.subplots(5,3, figsize=(25,25))

for i in range(15):

    j = i//3

    k = i%3

    ax[j,k].imshow(torch.transpose(a[0][i],0,2))

    ax[j,k].set_title(list(classes.keys())[a[1][i]])

plt.show()
submission = pd.DataFrame(test_samples.fname.str.split('/').str.get(-1))
#decode = {'daisy':1 ,'dandelion':2, 'rose':3,'sunflower':4,'tulip':5}

#decode = {'daisy':1 ,'dandelion':2, 'rose':3,'sunflower':4,'tulip':5}

decode = {0:1,1:2,2:3,3:4,4:5}

submission['label'] = prediction

submission.label = submission.label.map(decode)
submission
submission.to_csv('submission.csv',index=False)