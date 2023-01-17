!pip install wtfml==0.0.2
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from itertools import product
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from wtfml.data_loaders.image import ClassificationLoader
import albumentations

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# create 5 stratified k-folds in training set
melanoma_path="../input/siim-isic-melanoma-classification/"
melanoma_image_path="../input/siic-isic-224x224-images/"

df = pd.read_csv(melanoma_path + "train.csv")
df["kfold"] = -1    
df = df.sample(frac=1).reset_index(drop=True)
y = df.target.values
kf = model_selection.StratifiedKFold(n_splits=5)

for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f

# Create train and validation indices
df_train = df[df.kfold != 0].reset_index(drop=True)
df_valid = df[df.kfold == 0].reset_index(drop=True)
training_data_path=melanoma_image_path + "train/"
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_images = df_train.image_name.values.tolist()
train_images = [os.path.join(training_data_path, i + ".png") for i in train_images]
train_targets = df_train.target.values

train_aug = albumentations.Compose([
    albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
#     albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
#     albumentations.Flip(p=0.5)
])

train_dataset = ClassificationLoader(
    image_paths=train_images,
    targets=train_targets,
    resize=None,
    augmentations=train_aug,
)
valid_images = df_valid.image_name.values.tolist()
valid_images = [os.path.join(training_data_path, i + ".png") for i in valid_images]
valid_targets = df_valid.target.values

valid_aug = albumentations.Compose([
    albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
])

valid_dataset = ClassificationLoader(
    image_paths=valid_images,
    targets=valid_targets,
    resize=None,
    augmentations=valid_aug,
)
test_data_path = melanoma_image_path + "test/"
df_test = pd.read_csv(melanoma_path + "test.csv")

test_aug = albumentations.Compose([
        albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
])

test_images = df_test.image_name.values.tolist()
test_images = [os.path.join(test_data_path, i + ".png") for i in test_images]
test_targets = np.zeros(len(test_images))

test_dataset = ClassificationLoader(
    image_paths=test_images,
    targets=test_targets,
    resize=None,
    augmentations=test_aug,
)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=9, padding=4)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=18, kernel_size=5, padding=2)
#         self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=7, padding=3)
#         self.conv4 = nn.Conv2d(in_channels=12, out_channels=15, kernel_size=5, padding=2)
#         self.conv6 = nn.Conv2d(in_channels=18, out_channels=24, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=18*7*7, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=20)
        self.out = nn.Linear(in_features=20, out_features=2)
        
    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))  # Layer 1
        x = F.relu(self.pool1(self.conv3(x)))  # Layer 2
        x = F.relu(self.pool2(self.conv5(x)))  # Layer 3
        x = x.reshape(-1, 18*7*7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        
        return x
shuffle=True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

parameters = OrderedDict(
    batch_size=[100],
    lr = [0.01],
)

param_values = [v for v in parameters.values()]
print(param_values)
%%time

for batch_size, lr in product(*param_values):
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
#     trainloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, sampler = train_sampler)
#     testloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, sampler = valid_sampler)

    net = Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    comment = f'melanoma batch_size={batch_size} lr={lr}'
    print(comment)
#     tb = SummaryWriter(comment=comment)
#     tb_count=0

    for epoch in range(6): 
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['image'].to(device), data['targets'].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49:   
#                 tb_count += 1
#                 tb.add_scalar('Running Loss', running_loss/100, tb_count)
                print('[%d, %5d] loss: %.5f' %(epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

        if epoch % 2 == 1:
            print('At the end of epoch %d' %(epoch+1))
            correct = 0
            total = 0
            with torch.no_grad():
                preds=[]
                targets=[]
                for data in train_loader:
                    images, labels = data['image'].to(device), data['targets'].to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    preds += list(predicted.cpu().detach().numpy().squeeze())
                    targets += list(labels.cpu().detach().numpy().squeeze())
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

    #             tb.add_scalar('Train Accuracy', 100 * correct / total, epoch+1)
            print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))
            print('Training Confusion Matrix:')
            print(confusion_matrix(targets, preds))

            with torch.no_grad():
                for data in valid_loader:
                    images, labels = data['image'].to(device), data['targets'].to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    preds += list(predicted.cpu().detach().numpy().squeeze())
                    targets += list(labels.cpu().detach().numpy().squeeze())
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
    #             tb.add_scalar('Test Accuracy', 100 * correct / total, epoch+1)
            print('Accuracy of the network on the validation images: %d %%' % (100 * correct / total))
            print('Validation Confusion Matrix:')
            print(confusion_matrix(targets, preds))

#     tb.close()
    print('Finished Training')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=4)
with torch.no_grad():
    preds=[]
    targets=[]
    for data in train_loader:
        images, labels = data['image'].to(device), data['targets'].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        preds += list(predicted.cpu().detach().numpy().squeeze())
        targets += list(labels.cpu().detach().numpy().squeeze())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# tb.add_scalar('Train Accuracy', 100 * correct / total, epoch+1)
print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))
print(len(preds), len(targets))
print('Training Confusion Matrix:')
print(confusion_matrix(targets, preds))

