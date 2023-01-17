import numpy as np

import pandas as pd

import os



import cv2

import matplotlib.pyplot as plt

%matplotlib inline
labels = pd.read_csv('../input/aerial-cactus-identification/train.csv')

sub = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')
!mkdir "/kaggle/data/"

!mkdir "/kaggle/data/aerial-cactus-identification"

!unzip "/kaggle/input/aerial-cactus-identification/train.zip" -d "/kaggle/data/aerial-cactus-identification/train" > /dev/null

!unzip "/kaggle/input/aerial-cactus-identification/test.zip" -d "/kaggle/data/aerial-cactus-identification/test" > /dev/null



train_dir = "/kaggle/data/aerial-cactus-identification/train/train"

test_dir = "/kaggle/data/aerial-cactus-identification/test/test"
import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, Dataset

import torchvision

import torchvision.transforms as transforms



from sklearn.model_selection import train_test_split
## Parameters for model



# Hyper parameters

num_epochs = 5

batch_size = 128

learning_rate = 0.002



# Device configuration

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class MyDataset(Dataset):

    def __init__(self, df_data, data_dir = './', transform=None):

        super().__init__()

        self.df = df_data.values

        self.data_dir = data_dir

        self.transform = transform



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):

        img_name,label = self.df[index]

        img_path = os.path.join(self.data_dir, img_name)

        image = cv2.imread(img_path)

        if self.transform is not None:

            image = self.transform(image)

        return image, label # return transformed image and label
train, val = train_test_split(labels, stratify=labels.has_cactus, test_size=0.1) # 데이터를 has_cactus 레이블이 적당히 섞이도록 train과 val로 나누어준다.
trans_train = transforms.Compose([transforms.ToPILImage(),

                                  transforms.Pad(32, padding_mode='reflect'), # 32 paddings in each edge, reflect means [1,2,3] to [3,2,1,2,3,2,1]

                                  transforms.ToTensor(), # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]

                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])]) # 평균과 표준편차를 0.5로 잡고 정규화



trans_valid = transforms.Compose([transforms.ToPILImage(),

                                  transforms.Pad(32, padding_mode='reflect'),

                                  transforms.ToTensor(),

                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])



# Data generators

dataset_train = MyDataset(df_data=train, data_dir=train_dir, transform=trans_train)

dataset_valid = MyDataset(df_data=val, data_dir=train_dir, transform=trans_valid)



# num_workers=0 : 데이터를 메인 프로세스로 불러옴.

loader_train = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True, num_workers=0) # (C x H x W)의 이미지를 batch_size 갯수만큼 묶어 (128, C, H, W)의 텐서로 만들어 준다.

loader_valid = DataLoader(dataset = dataset_valid, batch_size=batch_size//2, shuffle=False, num_workers=0)
# NOTE: class is inherited from nn.Module

class SimpleCNN(nn.Module):

    def __init__(self):

        super(SimpleCNN, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3, 64, 3, padding=1),nn.LeakyReLU(0.2), # LeakyReLU(x)=max(0,x)+negative_slope∗min(0,x)

            nn.Conv2d(64, 64, 3, padding=1),nn.LeakyReLU(0.2),

            nn.MaxPool2d(2, 2),



            nn.Conv2d(64, 128, 3, padding=1),nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, 3, padding=1),nn.LeakyReLU(0.2),

            nn.MaxPool2d(2, 2),



            nn.Conv2d(128, 256, 3, padding=1),nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),

            nn.MaxPool2d(2, 2),



            nn.Conv2d(256, 512, 3, padding=1),nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),

            nn.MaxPool2d(2, 2),



            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),

            nn.MaxPool2d(2, 2) # conv의 결과는 (128, 3, 3, 512) 크기의 텐서, 128=batch_size, (3,3)=height,width, 512=channels

        )



        self.classifier = nn.Linear(3*3*512, 2) # 결과가 2개이므로 2, 우리는 3을 사용할 예정

   

    def forward(self, x):

        x = self.conv(x)

        

        x = x.view(-1, 3*3*512) # view means reshape

        x = self.classifier(x) # don't use many FC layers to enhance calculation speed

        

        return x
model = SimpleCNN().to(device)
# Loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
# Train the model

total_step = len(loader_train) # 전체 데이터 갯수 / 배치 사이즈 = 1 에폭 당 돌려야 하는 배치의 갯수

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(loader_train):

        images = images.to(device)

        labels = labels.to(device)

        

        # Forward pass

        outputs = model(images)

        loss = criterion(outputs, labels)

        

        # Backward and optimize

        optimizer.zero_grad() # zero the gradient

        loss.backward() # 역전파 단계: 모델의 매개변수에 대한 손실의 변화도를 계산합니다.

        optimizer.step() # Optimizer의 step 함수를 호출하면 매개변수가 갱신됩니다.

        

        if (i+1) % 100 == 0:

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 

                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
# Test the model

# model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time.

# For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you.

model.eval()

with torch.no_grad(): # validation inner loop를 이 context manager로 감싸서 gradient calculation을 방지하는 것이 좋다.

    correct = 0

    total = 0

    for images, labels in loader_valid:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1) # torch.max(input, dim) returns (values, indices) in this case it returns the indicies of column which has max value (ex) [0, 1, 1, 0, ...]

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

          

    print('Test Accuracy of the model on the 1750 validation images: {} %'.format(100 * correct / total))



# Save the model checkpoint

torch.save(model.state_dict(), 'model.ckpt')
# generator for test data 

dataset_valid = MyDataset(df_data=sub, data_dir=test_dir, transform=trans_valid)

loader_test = DataLoader(dataset = dataset_valid, batch_size=32, shuffle=False, num_workers=0)
model.eval()



preds = []

for batch_i, (data, target) in enumerate(loader_test):

    data, target = data.cuda(), target.cuda()

    output = model(data)



    pr = output[:,1].detach().cpu().numpy()

    for i in pr:

        preds.append(i)



sub['has_cactus'] = preds

sub.to_csv('sub.csv', index=False)