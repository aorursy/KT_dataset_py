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



# Any results you write to the current directory are saved as output
import torch

import torchvision

import torch.optim as optim

import torch.nn as nn

from torchvision import transforms, models

from torch.utils.data import DataLoader, Dataset, ConcatDataset

import torch.nn.functional as F

from PIL import Image

import matplotlib.pyplot as plt

import os

import glob

from sklearn.model_selection import train_test_split
#GPU設定

device = "cuda" if torch.cuda.is_available() else "cpu"

device
#zipファイルを解凍

!unzip "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
#trainの中身を確認

!ls train
#test

!unzip "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"
#中身を確認

!ls test
train_dir = 'train'

test_dir = 'test'
#ファイルをリスト化

train_list = glob.glob(os.path.join(train_dir, '*.jpg'))

test_list = glob.glob(os.path.join(test_dir, '*jpg'))
#件数を確認

print('train_list:{}'.format(len(train_list)))

print('test_list:{}'.format(len(test_list)))
#データセットを確認

random_idx = np.random.randint(1, 25000, size=10)

fig = plt.figure()

i = 1

figsize = (15, 15)

for idx in random_idx:

    figsize = figsize

    ax = fig.add_subplot(2,5,i)

    img = Image.open(train_list[idx])

    plt.imshow(img)

    i += 1

    plt.axis('off')

plt.show()

#ファイルの名前のtrainとtestの違い

print('train_file_name:{}'.format(train_list[0].split('/')[-1].split('.')[0]))

print('test_file_name:{}'.format(test_list[0].split('/')[-1].split('.')[0]))

#trainは写っているのがdogかcatか

#testは番号のみ
#trainを学習用と検証用に分ける

train_list, val_list = train_test_split(train_list, test_size=0.2)
#前処理の定義

#train

train_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomResizedCrop(224),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor()

])



#validation

val_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomResizedCrop(224),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    

])



#test

test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    #transforms.RandomResizedCrop(224),

    #transforms.RandomHorizontalFlip(),

    transforms.ToTensor()

])
#カスタムデータセット定義

class dataset(Dataset):

    #初期化

    def __init__(self, file_list, transform=None):

        self.file_list = file_list

        self.transform = transform

    

    #特殊メソッド

    def __len__(self):

        return len(self.file_list)

    

    def __getitem__(self, idx):

        img_path = self.file_list[idx]

        img = Image.open(img_path)

        img_transformed = self.transform(img)

        

        #ラベルの定義

        label = img_path.split('/')[-1].split('.')[0]

        if label == 'dog':

            label = 1

        else:

            label = 0   

        return img_transformed, label, img_path
#データセット作成

train_data = dataset(train_list, transform=train_transform)

val_data = dataset(val_list, transform=val_transform)

test_data = dataset(test_list, transform=test_transform)
#dataloaderを使ってバッチ処理

train_loader = DataLoader(dataset = train_data, batch_size=100, shuffle=True)

test_loader = DataLoader(dataset = test_data, shuffle=False)

val_loader = DataLoader(dataset = val_data, batch_size=100, shuffle=True)
print(len(train_data), len(train_loader))
print(len(val_data), len(val_loader))
print(len(test_data), len(test_loader))
#画像のshapeを確認

train_data[0][0].shape
class CNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.layer1 = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2)

        )

        self.layer2 = nn.Sequential(

            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2)

        )

        self.layer3 = nn.Sequential(

            nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2)

        )

        self.fc1 = nn.Linear(3*3*64, 10)

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(10, 2)

        self.relu = nn.ReLU()

    

    #順伝播

    def forward(self, x):

        out = self.layer1(x)

        out = self.layer2(out)

        out = self.layer3(out)

        out = out.view(out.size(0), -1)

        out = self.relu(self.fc1(out))

        out = self.fc2(out)

        return out

        

        
#インスタンス作成

model = CNN(2)

#GPUに送る

model.to(device)

#??

model.train()
#損失関数とoptimizerの設定

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
#データが正しく取得できているか確認

data_iter = iter(train_loader)

data, label, path = data_iter.next()

label
test_iter = iter(test_loader)

test_data, _, test_path = test_iter.next()

test_path = str(test_path)

test = []

for test_data, _, test_path in test_loader:

    test_path = str(test_path)

    idx = [test_path.split('/')[-1].split('.')[0]]

    print(idx)

    test += list(zip(idx, test_data))

test
epochs = 15

losses = []

acces = []

val_losses = []

val_acces = []

for epoch in range(epochs):

    epoch_loss = 0

    epoch_acc = 0  

    for data, label,_ in train_loader:

        data = data.to(device)

        label = label.to(device)

        output = model(data)

        loss = criterion(output, label)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        acc = ((output.argmax(dim=1) == label).float().mean())

        epoch_acc += acc/len(train_loader)

        epoch_loss += loss/len(train_loader)

        losses.append(epoch_loss)

        acces.append(epoch_acc)

    print('Epoch:{}, train_acc:{}, train_loss:{}'.format(epoch+1, epoch_acc, epoch_loss))

    

    #val_loop

    with torch.no_grad():

        epoch_val_acc = 0

        epoch_val_loss = 0

        for data, label, _ in val_loader:

            data = data.to(device)

            label = label.to(device)

            val_output = model(data)

            val_loss = criterion(val_output, label)

            acc = ((val_output.argmax(dim=1)==label).float().mean())

            epoch_val_acc += acc/len(val_loader)

            epoch_val_loss += val_loss/ len(val_loader)

            val_losses.append(epoch_val_loss)

            val_acces.append(epoch_val_acc)

    print('Epoch:{}, val_accuracy:{}, val_loss:{}'.format(epoch+1, epoch_val_acc, epoch_val_loss))
plt.style.use("ggplot")

plt.plot(losses, label="train_loss")

plt.plot(val_losses, label="validation_loss")

plt.legend()
plt.plot(acces, label="train_acc")

plt.plot(val_acces, label="validation_acc")

plt.legend()
#test



dog_probs = []

model.eval()



with torch.no_grad():

    for test_data, test_label, test_file in test_loader:

        data = test_data.to(device)

        preds = model(data)

        preds_list = F.softmax(preds, dim=1)[:, 1].tolist()

        test_file = str(test_file)

        idx = [test_file.split('/')[-1].split('.')[0]]

        #print(idx)

        dog_probs += list(zip(idx, preds_list))

    dog_probs

    

        
dog_probs.sort(key = lambda x : int(x[0]))

dog_probs
idx = list(map(lambda x: x[0], dog_probs))

prob = list(map(lambda x: x[1], dog_probs))
submission = pd.DataFrame({'id': idx, 'label':prob})

submission
submission.to_csv('result.csv', index=False)
#モデルの精度を可視化してして見てみる

import random

id_list = []

class_ = {0: 'cat', 1:'dog'}

fig, axes = plt.subplots(2, 5, figsize=(20, 12), facecolor='w')

for ax in axes.ravel():

    i = random.choice(submission['id'].values)

    label = submission.loc[submission['id'] == i, 'label'].values[0]

    if label > 0.5:

        label = 1

    else:

        label = 0

    img_path = os.path.join(test_dir, '{}.jpg'.format(i))

    img = Image.open(img_path)

    

    ax.set_title(class_[label])

    ax.imshow(img)

        