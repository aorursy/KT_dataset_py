!pip3 install torchsummary

import torch
from torch import nn, optim
from torch.nn import Module as M
from torch.utils.data import Dataset as D
# import torchvision
from torchvision import datasets # for mnist
import torchvision.transforms as transforms
from torchsummary import summary

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os
import numpy as np
import random
import math
from tqdm import tqdm

import albumentations
#from albumentations.pytorch import ToTensorV2 as AT

import cv2

#from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
#!ls /kaggle/input/digit-recognizer
IMG_SIZE = 28
epoch_count = 5
fold_count = 5
SEED = 42

batch_size = 64

# Скорость обучения
LR = 5e-5

# Параметры оптимизатора Adam
beta1 = 0.9
beta2 = 0.999

base_patch = '/kaggle/input/digit-recognizer'
train_file = os.path.join(base_patch, "train.csv")
test_file = os.path.join(base_patch, "test.csv")
submission_file = os.path.join(base_patch, "sample_submission.csv")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
submission_df = pd.read_csv(submission_file)
submission_df.iloc[:, 1:] = 0

submission_df.head()

class digitModel(M):
    def __init__(self):
        super(digitModel, self).__init__()
        # Формeлf расчета размера выходного слоя после Conv2d
        # c_out = ((c_in+2pading-kernel_size)/strides)+1
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1) # 28
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # 14
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0) # 12
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # 6
        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0) # 4
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)  # 2
        # Convolution 4
        self.cnn4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # 2
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)  # 1
        # Fully connected 1
        self.fc1 = nn.Linear(128 * 1 * 1, 10)

    def forward(self, x):
        # Convolution 1
        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # Convolution 2
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # Convolution 3
        x = self.cnn3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        # Convolution 4
        x = self.cnn4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        # подготовка для линейного слоя
        x = x.view(x.size(0), -1)
        # Linear function (readout)
        x = self.fc1(x)
        return x
class digitDataset(D):
    def __init__(self, df, transform=None): #, labels=None
        if 'label' in df:
            self.labels = df['label'].values
            self.images = df.drop(axis=1, columns='label')
        else:
            self.labels = np.zeros(len(df))
            self.images = df
        
        # Нормализуем
        self.images = np.multiply(np.array(self.images, dtype=np.float32),1/255)
        self.images = self.images.reshape(-1,1,28,28)
        self.images = torch.from_numpy(self.images)

        self.transform = transform
            
    def __len__(self):
        return len(self.images)
        #return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]
        
        #применяем аугментации
        if self.transform:
            image = self.transform(image)
            
        return image, label


def imshow(imgs, lbls, epoh='', batch=''):
    fig = plt.figure(figsize=(10, 11))
    for j in range(batch_size):#len(data)):
        n = math.sqrt(batch_size)
        ax = fig.add_subplot(n, n, j+1)
        #i = random.randrange(0,batch_size)
        ax.set_title(str(lbls[j].numpy()))
        ax.imshow(imgs[j].reshape(28,28), cmap = cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()
    plt.close()
model = digitModel().to(device)
#optimiser = optim.SGD(model.parameters(), lr=LR,)
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(beta1, beta2))
criterion = nn.CrossEntropyLoss()
summary(model, (1,28,28))
def train(model, train_loader, criterion, optimizer, show=False):
    model.train()
    tr_loss = 0
    
    for step, batch in enumerate(tqdm(train_loader)):
        images = batch[0]
        labels = batch[1]
        
        if show:
            imshow(images, labels);
        
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels.squeeze(-1))                
        loss.backward()

        tr_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()
        
    return tr_loss / len(train_loader)
def valid(model, valid_loader, criterion, optimizer):
    model.eval()
    val_loss = 0
    correct = 0
    count = 0

    for step, batch in enumerate(tqdm(valid_loader)):

            images = batch[0]
            labels = batch[1]

            count += len(images)

            if val_labels is None:
                val_labels = labels.clone()
            else:
                val_labels = torch.cat((val_labels, labels), dim=0)

            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                    
    return val_loss / len(train_loader), correct / count
def test(model, test_loader, criterion, optimizer):

    test_preds = None
    
    model.eval()
    test_preds = None
    
    for step, batch in enumerate(tqdm(test_loader, ncols=80, position=0)):

        images = batch[0]
        images = images.to(device) #, dtype=torch.float)

        with torch.no_grad():
            outputs = model(images)

            _, predicted = torch.max(outputs.data.cpu(), 1)
            if test_preds is None:
                test_preds = predicted
            else:
                test_preds = torch.cat((test_preds, predicted), dim=0)
    return test_preds

train_transform = transforms.Compose([
    transforms.ToPILImage(), # преобрахование из numpi.array в PILImage
    #transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    ##transforms.Resize(32, interpolation=2), # увеличиваем для RandomRotation
    #transforms.RandomRotation(degrees=(-10, 10), expand=True, fill=(0,)),
    #transforms.Resize(IMG_SIZE, interpolation=2), #
    ##transforms.Normalize(mean=[0.456],
    ##                     std=[0.224]),
    transforms.ToTensor(),
    ])

test_transform = transforms.Compose([
    transforms.ToPILImage(), # преобрахование из numpi.array в PILImage
    #transforms.Normalize(mean=[0.456],
    #                     std=[0.224]),
    transforms.ToTensor(),
    ])

data = pd.read_csv(train_file)

#folds = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=SEED)
folds = KFold(n_splits=fold_count, shuffle=True, random_state=SEED)

# MNIST
kwargs = {} #{'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader_MNIST = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
test_loader_MNIST = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

#for i_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_y)):
for i_fold, (train_idx, valid_idx) in enumerate(folds.split(data)):
    train_data = data.iloc[train_idx]
    train_data.reset_index(drop=True, inplace=True)

    valid_data = data.iloc[valid_idx]
    valid_data.reset_index(drop=True, inplace=True)
    

    #Инициализируем датасеты

    trainset = digitDataset(train_data, transform=train_transform)
    validset = digitDataset(valid_data, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(trainset, 
                                           batch_size = batch_size, 
                                           shuffle = True)
    valid_loader = torch.utils.data.DataLoader(validset, 
                                          batch_size = batch_size, 
                                          shuffle = False)
    
    for epoch in range(epoch_count):
        tr_loss = train(model, train_loader, criterion, optimizer, show=False)
        val_loss, predicted = valid(model, valid_loader, criterion, optimizer)
        print(i_fold+1, '-', epoch+1, tr_loss, val_loss, predicted)
        
        
        #MNIST
        tr_loss = train(model, train_loader_MNIST, criterion, optimizer, show=False)
        #val_loss, predicted = valid(model, valid_loader, criterion, optimizer)
        #print('MNIST',i_fold+1, '-', epoch+1, tr_loss, val_loss, predicted)

        #tr_loss = train(model, test_loader_MNIST, criterion, optimizer)
        val_loss, predicted = valid(model, test_loader_MNIST, criterion, optimizer)
        #val_loss, predicted = valid(model, valid_loader, criterion, optimizer)
        print('MNIST',i_fold+1, '-', epoch+1, tr_loss, val_loss, predicted)
        
    
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': tr_loss,
                    },
                   'model.pth.tar',
                   )

test_data = pd.read_csv(test_file)


testset = digitDataset(test_data, transform=test_transform)

test_loader = torch.utils.data.DataLoader(testset, 
                                          batch_size = batch_size, 
                                          shuffle = False)

test_preds = test(model, test_loader, criterion, optimizer)

submission_df['Label'] = test_preds
submission_df.to_csv('submission.csv', index=False)

