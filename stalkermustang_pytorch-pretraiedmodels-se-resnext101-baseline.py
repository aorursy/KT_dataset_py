!pip install albumentations > /dev/null
!pip install pretrainedmodels > /dev/null
import os

import albumentations
from albumentations import torch as AT
import pretrainedmodels

import numpy as np
import pandas as pd

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from tqdm import tqdm

from matplotlib import pyplot as plt
%matplotlib inline

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
train_df = pd.read_csv("../input/train.csv")
train_df.head()
train_df.shape, train_df.Id.nunique()
NUM_CLASSES = train_df.Id.nunique()
train_df.Id.value_counts().iloc[1:].hist(bins=40)
RESIZE_H = 160
RESIZE_W = 320

data_transforms = albumentations.Compose([
    albumentations.Resize(RESIZE_H, RESIZE_W),
    albumentations.HorizontalFlip(),
    albumentations.OneOf([
        albumentations.RandomContrast(),
        albumentations.RandomBrightness(),
    ]),
    albumentations.ShiftScaleRotate(rotate_limit=10, scale_limit=0.15),
    albumentations.JpegCompression(80),
    albumentations.HueSaturationValue(),
    albumentations.Normalize(),
    AT.ToTensor()
])

data_transforms_test = albumentations.Compose([
    albumentations.Resize(RESIZE_H, RESIZE_W),
    albumentations.Normalize(),
    AT.ToTensor()
])
def prepare_labels(y):
    # From here: https://www.kaggle.com/pestipeti/keras-cnn-starter
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    y = onehot_encoded
    return y, label_encoder
y, lab_encoder = prepare_labels(train_df['Id'])
class WhaleDataset(Dataset):
    def __init__(self, datafolder, datatype='train', df=None, transform=None, y=None):
        self.datafolder = datafolder
        self.datatype = datatype
        self.y = y
        if self.datatype == 'train':
            self.df = df.values
        self.image_files_list = [s for s in os.listdir(datafolder)]
        self.transform = transform


    def __len__(self):
        return len(self.image_files_list)
    
    def __getitem__(self, idx):
        if self.datatype == 'train':
            img_name = os.path.join(self.datafolder, self.df[idx][0])
            label = self.y[idx]
            
        elif self.datatype == 'test':
            img_name = os.path.join(self.datafolder, self.image_files_list[idx])
            label = np.zeros((NUM_CLASSES,))

        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = self.transform(image=img)['image']
        if self.datatype == 'train':
            return image, label
        elif self.datatype == 'test':
            # so that the images will be in a correct order
            return image, label, self.image_files_list[idx]
train_dataset = WhaleDataset(
    datafolder='../input/train/', 
    datatype='train', 
    df=train_df, 
    transform=data_transforms, 
    y=y
)

test_set = WhaleDataset(
    datafolder='../input/test/', 
    datatype='test', 
    transform=data_transforms_test
)
batch_size = 10
num_workers = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
model = pretrainedmodels.resnext101_64x4d()
model.avg_pool = nn.AvgPool2d((5,10))
model.last_linear = nn.Linear(model.last_linear.in_features, NUM_CLASSES)
model.cuda();
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0005)

scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
def cuda(x):
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x
n_epochs = 5
for epoch in range(1, n_epochs+1):
    train_loss = []
    
    for batch_i, (data, target) in tqdm(enumerate(train_loader), total = len(train_loader)):
        data, target = cuda(data), cuda(target)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.float())
        train_loss.append(loss.item())

        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}')
sub = pd.read_csv('../input/sample_submission.csv')

model.eval()
for (data, target, name) in tqdm(test_loader):
    data = cuda(data)
    output = model(data)
    output = output.cpu().detach().numpy()
    for i, (e, n) in enumerate(list(zip(output, name))):
        sub.loc[sub['Image'] == n, 'Id'] = ' '.join(lab_encoder.inverse_transform(e.argsort()[-5:][::-1]))
        
sub.to_csv('submission.csv', index=False)