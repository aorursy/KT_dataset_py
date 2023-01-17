# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd
def convert(name):

    category = {'daisy':0, 'sunflower':1, 'tulip':2, 'rose':3, 'dandelion':4}

    return category[name]



def to_onehot(category, size=5):

    onehot = np.zeros(5)

    onehot[category] = 1

    return onehot

    

    

data = []
data_dir = '../input/flowers-recognition/flowers/'

for name in os.listdir(data_dir):

    if 'flowers' in name:

        continue

    for path in os.listdir(data_dir+name):

        if '.jpg' not in path:

            continue

        data.append([data_dir+name+'/'+path, name])
data_df = pd.DataFrame(data, columns=['path', 'category'])
data_df['type'] = list(map(convert, data_df['category']))
data_df['type'].plot.hist()

len(data_df.head(10))

data_df.head(10)
data_df['path'][0]
data_df.to_csv('flowers.csv', index=False)
import torch

import torch.utils.data as data

import cv2

import matplotlib.pyplot as plt



from albumentations import CenterCrop, ShiftScaleRotate, RandomCrop, HorizontalFlip, VerticalFlip, RandomSizedCrop, Blur, OneOf, Compose, GaussianBlur, MotionBlur, MedianBlur, RandomBrightnessContrast, HueSaturationValue, Normalize  
def augment(is_train, p=1.0):

    if is_train:

        return Compose([

#             OneOf([

#                 GaussianBlur(p=0.5),

#                 Blur(blur_limit=3, p=0.5)

#             ]),

#             OneOf([

#                 MedianBlur(blur_limit=3, p=0.5),

#                 MotionBlur(blur_limit=3, p=0.5)

#             ]),

            RandomBrightnessContrast(p=0.5),

#             HueSaturationValue(p=0.5),

            ShiftScaleRotate(p=0.5),

            OneOf([

                HorizontalFlip(p=0.5),

                VerticalFlip(p=0.5)

            ]),

            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), p=1.0),

#             CentralCrop()

            

        ], p=p)

    else:

        return Compose([

            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), p=1.0),

        ], p=p)
class FlowerDataset(data.Dataset):

    def __init__(self, data_df, is_train, image_size, transforms=None):

        self.data_df = data_df

        self.is_train = is_train

        self.image_size = image_size

        self.transforms = transforms

        

        

    def __len__(self):

        return len(self.data_df) 

    

    def __getitem__(self, idx):

        path = self.data_df['path'][idx]

        category = self.data_df['type'][idx]

        image = cv2.imread(path)

        if self.transforms is not None:

            transformed = self.transforms(image=image, label=category)

            image, category = transformed['image'], transformed['label']

        image = cv2.resize(image, self.image_size)

        image = np.transpose(image, (2, 0, 1))

        image = torch.FloatTensor(image)

        category = torch.LongTensor([category])

        

        return image, category

        
dataset = FlowerDataset(data_df, True, (256, 256))
img, category = dataset[2]

category
# for img, category in dataset:

#     a = 0

len(dataset)

len(data_df)
from sklearn.model_selection import train_test_split



def get_train_test_dataset(data_df, train_size=0.8):

    train_df, test_df = train_test_split(data_df, test_size=1-train_size, stratify=data_df['type'])

    valid_df, test_df = train_test_split(test_df, test_size=1-0.5, stratify=test_df['type'])

    print(len(train_df), len(valid_df), len(test_df))

    train_df = train_df.reset_index()

    test_df = test_df.reset_index()

    valid_df = valid_df.reset_index()

    print(train_df.head())

    train_aug = augment(is_train=True, p=1.0)

    test_aug = augment(is_train=False, p=1.0)



    train_ds = FlowerDataset(train_df, True, (224,224), transforms=train_aug)

    test_ds = FlowerDataset(train_df, False, (224,224), transforms=test_aug)       

    valid_ds = FlowerDataset(valid_df, False, (224,224), transforms=test_aug)       

    return train_ds, valid_ds, test_ds
train_ds, valid_ds, test_ds = get_train_test_dataset(data_df, train_size=0.8)
len(train_ds)
# for _,_ in train_ds:

#     a = 0

    
train_loader = data.DataLoader(train_ds, shuffle=True, batch_size=32, num_workers=1)

valid_loader = data.DataLoader(valid_ds, shuffle=True, batch_size=32, num_workers=1)

test_loader = data.DataLoader(test_ds, shuffle=True, batch_size=32, num_workers=1)
import torch.nn as nn

import torch.nn.functional as F

import tqdm

from torch.optim.adam import Adam
class Conv(nn.Module):

    def __init__(self, n_in, n_out, kernel_size, stride, padding):

        super().__init__()

        self.conv = nn.Conv2d(n_in, n_out, kernel_size, stride, padding, bias=False)

        self.bn = nn.BatchNorm2d(n_out)

        self.relu = nn.ReLU(inplace=True)

    

    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        x = self.relu(x)

        return x

class VGG(nn.Module):

    def __init__(self, n_classes):

        super().__init__()

        self.n_classes = n_classes

        self.layer_1 = nn.Sequential(

                        Conv(3, 64, 3, 1, 1),

                        Conv(64, 64, 3, 1, 1)

                    )

        self.layer_2 = nn.Sequential(

                        Conv(64, 128, 3, 1, 1)

                    )

        self.layer_3 = nn.Sequential(

                        Conv(128, 256, 3, 1, 1),

                        Conv(256, 256, 3, 1, 1)

                    )

        self.layer_4 = nn.Sequential(

                        Conv(256, 512, 3, 1, 1),

                        Conv(512, 512, 3, 1, 1)

                    )

        self.layer_5 = nn.Sequential(

                        Conv(512, 512, 3, 1, 1),

                        Conv(512, 512, 3, 1, 1)

                    )

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(

                        nn.Linear(512, 128),

                        nn.Dropout(0.5),

                        nn.ReLU(inplace=True),

                        nn.Linear(128, self.n_classes)



                    )

    

    def forward(self, x):

        x = self.pooling(self.layer_1(x))

        x = self.pooling(self.layer_2(x))        

        x = self.pooling(self.layer_3(x))        

        x = self.pooling(self.layer_4(x))

        x = self.pooling(self.layer_5(x))

        x = self.global_pool(x).squeeze()

        x = self.classifier(x)

        return x
class ResBlock(nn.Module):

    def __init__(self, n_in, n_out, kernel_size, stride, padding):

        super().__init__()

        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size, stride, padding, bias=False)

        self.bn1 = nn.BatchNorm2d(n_out)

        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size, 1, 1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.bn2 = nn.BatchNorm2d(n_out)

        if n_in != n_out or stride != 1:

            self.shortcut = nn.Sequential(

                nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride),

                nn.BatchNorm2d(n_out)

            )

        else:

            self.shortcut=None

    

    def forward(self, x):

        residual = x

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.conv2(x)

        x = self.bn2(x)

        

        if self.shortcut is not None:

            residual = self.shortcut(residual)

        

        x += residual

        x = self.relu(x)

        

        return x

                

        
class ResNet18(nn.Module):

    def __init__(self, n_classes):

        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(

            ResBlock(64, 64, 3, 1, 1),

            ResBlock(64, 64, 3, 1, 1)

        ) 

        self.layer2 = nn.Sequential(

            ResBlock(64, 128, 3, 2, 1),

            ResBlock(128, 128, 3, 1, 1)

        )

            

        self.layer3 = nn.Sequential(

            ResBlock(128, 256, 3, 2, 1),

            ResBlock(256, 256, 3, 2, 1)

        ) 

        self.layer4 = nn.Sequential(

            ResBlock(256, 512, 3, 2, 1),

            ResBlock(512, 512, 3, 1, 1)

        ) 

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, n_classes)

        

    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x).squeeze()

        x = self.fc(x)

        

        return x

        
def train(model, epoch, train_loader, valid_loader, loss_fn, acc_metric, optimizer, device):

    for i in range(epoch):

        print(f'epoch {i+1}/{epoch}')

        model = model.to(device)

        # train

        model.train()

        losses = []

        acc = 0

        for train_input, train_label in (train_loader):

            train_input, train_label = train_input.to(device), train_label.to(device).squeeze()

            train_output = model(train_input)

            loss = loss_fn(train_output, train_label)



            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



            losses.append(loss.item())

            acc += acc_metric(train_output, train_label)

        

        print(acc, len(train_loader))

        acc = (acc / len(train_loader)) * 100



        print(f'train loss: {np.mean(losses)} and train acc: {acc}')



        model.eval()

        eval_losses = []

        eval_acc = 0

        for eval_input, eval_label in (valid_loader):

            eval_input, eval_label = eval_input.to(device), eval_label.to(device).squeeze()

            eval_output = model(eval_input)

            loss = loss_fn(eval_output, eval_label)



            eval_losses.append(loss.item())

            eval_acc += acc_metric(eval_output, eval_label)



        eval_acc = eval_acc / len(valid_loader) * 100

        print(f'valid loss: {np.mean(eval_losses)} and valid acc: {eval_acc}')

        print('*******')



        

def acc_metric(pred, label):

    pred = torch.nn.functional.softmax(pred, dim=1)

    pred = torch.argmax(pred, dim=1).cpu().numpy()

    label = label.cpu().numpy()

    

    acc = np.array((label == pred)).sum()/len(label)

    return acc
model = VGG(n_classes=5)

model



# model = ResNet18(n_classes=5)

# model
loss_fn = nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=1e-5)

if torch.cuda.is_available():

    device = torch.device('cuda')

else:

    device = torch.device('cpu')

device
train(model, 100, train_loader, valid_loader, loss_fn, acc_metric, optimizer, device)