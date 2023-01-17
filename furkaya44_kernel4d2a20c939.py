import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set(style="darkgrid")
from PIL import Image

import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
train_images_folder_path = "../input/histopathologic-cancer-detection/train"
test_images_folder_path = "../input/histopathologic-cancer-detection/test"
labels_file_path = '../input/histopathologic-cancer-detection/train_labels.csv'
labels = pd.read_csv(labels_file_path)
labels.head()
fig, ax = plt.subplots(figsize=(5,5))
ax.pie([labels.label.value_counts()[0], labels.label.value_counts()[1]], labels=['Negative', 'Positive'], autopct='%1.1f%%');
sns.countplot(x='label', data=labels);
fig, ax = plt.subplots(2,5, figsize=(20,8))
fig.suptitle('Histopathologic scans of lymph node sections')

# Negative Samples
for i, image_id in enumerate(labels[labels.label == 0]['id'][:5]):
    path = os.path.join(train_images_folder_path, image_id)
    ax[0,i].imshow(Image.open(path + '.tif'))
    box = patches.Rectangle((32,32), 32, 32, linewidth=5, edgecolor='g', facecolor='none')
    ax[0,i].add_patch(box)
ax0 = ax[0,0].set_ylabel('Negative samples')

# Positive Samples
for i, image_id in enumerate(labels[labels.label == 1]['id'][:5]):
    path = os.path.join(train_images_folder_path, image_id)
    ax[1,i].imshow(Image.open(path + '.tif'))
    box = patches.Rectangle((32,32), 32, 32, linewidth=5, edgecolor='r', facecolor='none')
    ax[1,i].add_patch(box)
ax1 = ax[1,0].set_ylabel('Positive samples')
train_indices, validation_indices = train_test_split(labels.label, stratify=labels.label, test_size=0.3)
data_transformations_train = transforms.Compose([
    transforms.Pad(64, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_transformations_test = transforms.Compose([
    transforms.Pad(64, padding_mode='reflect'),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
fig, ax = plt.subplots(1,6, figsize=(24,4))

image_path = os.path.join(train_images_folder_path, 'c18f2d887b7ae4f6742ee445113fa1aef383ed77.tif')
image_original = Image.open(image_path)

center_crop = transforms.CenterCrop(64)

composition = transforms.Compose([
    transforms.Pad(64, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15)
])

ax[0].imshow(image_original)
ax[1].imshow(center_crop(image_original))
ax[2].imshow(transforms.functional.hflip(image_original))
ax[3].imshow(transforms.functional.vflip(image_original))
ax[4].imshow(transforms.functional.rotate(image_original, 15))
ax[5].imshow(composition(image_original))

ax0 = ax[0].set_xlabel('Original image')
ax1 = ax[1].set_xlabel('Pad')
ax2 = ax[2].set_xlabel('Horizontal Flip')
ax3 = ax[3].set_xlabel('Vertical Flip')
ax4 = ax[4].set_xlabel('Rotation')
ax5 = ax[5].set_xlabel('Composition of these transformations')
class PCamDataset(Dataset):
    def __init__(self, data_folder, data_type, transform, labels_dict={}):
        self.data_folder = data_folder
        self.data_type = data_type
        self.image_files_list = [image_file_name for image_file_name in os.listdir(data_folder)]
        self.transform = transform
        self.labels_dict = labels_dict
        if self.data_type == 'train':
            self.labels = [labels_dict[image_file_name.split('.')[0]] for image_file_name in self.image_files_list]
        else:
            self.labels = [0 for _ in range(len(self.image_files_list))]

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, index):
        image_file_name = os.path.join(self.data_folder, self.image_files_list[index])
        image = Image.open(image_file_name)
        image = self.transform(image)
        image_id = self.image_files_list[index].split('.')[0]
        if self.data_type == 'train':
            label = self.labels_dict[image_id]
        else:
            label = 0
        return image, label
image_labels_dict = { image:label for image, label in zip(labels.id, labels.label) }

dataset = PCamDataset(data_folder=train_images_folder_path, data_type='train', transform=data_transformations_train, labels_dict=image_labels_dict)
test_set = PCamDataset(data_folder=test_images_folder_path, data_type='test', transform=data_transformations_test)

train_sampler = SubsetRandomSampler(list(train_indices.index))
valid_sampler = SubsetRandomSampler(list(validation_indices.index))

batch_size = 64

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
test_loader = DataLoader(test_set, batch_size=batch_size)
model = torchvision.models.resnet18(pretrained=True)
for i, param in model.named_parameters():
    param.requires_grad = False
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
n_epochs = 4
patience = 5
p = 0
stop = False
valid_loss_min = np.Inf

train_loss_epoch = []
val_loss_epoch = []
val_auc_epoch = []


for epoch in range(1, n_epochs+1):

    train_loss = []
    train_auc = []

    for batch_i, (data, target) in enumerate(train_loader):

        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output[:,1], target.float())
        train_loss.append(loss.item())
        
        a = target.data.cpu().numpy()
        b = output[:,-1].detach().cpu().numpy()
        train_auc.append(roc_auc_score(a, b))

        loss.backward()
        optimizer.step()
    
    exp_lr_scheduler.step()
    
    train_loss_epoch.append(np.mean(train_loss))
    
    model.eval()
    
    val_loss = []
    val_auc = []
    
    for batch_i, (data, target) in enumerate(valid_loader):
        data, target = data.cuda(), target.cuda()
        output = model(data)

        loss = criterion(output[:,1], target.float())

        val_loss.append(loss.item()) 
        a = target.data.cpu().numpy()
        b = output[:,-1].detach().cpu().numpy()
        val_auc.append(roc_auc_score(a, b))

    val_loss_epoch.append(np.mean(val_loss))
    val_auc_epoch.append(np.mean(val_auc))
    
    print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}, valid loss: {np.mean(val_loss):.4f}, train auc: {np.mean(train_auc):.4f}, valid auc: {np.mean(val_auc):.4f}')
    
    valid_loss = np.mean(val_loss)
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = valid_loss
        p = 0

    # check if validation loss didn't improve
    if valid_loss > valid_loss_min:
        p += 1
        print(f'{p} epochs of increasing val loss')
        if p > patience:
            print('Stopping training')
            stop = True
            break        
    if stop:
        break
model = torchvision.models.resnet18(pretrained=True)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model.cuda()
model.load_state_dict(torch.load('model.pt'))
model.eval();
plt.plot(train_loss_epoch, label='Train Loss')
plt.plot(val_loss_epoch, label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend();
plt.plot(val_auc_epoch, label='Validation AUC')
plt.xlabel("Epochs")
plt.ylabel("Area Under the Curve")
plt.legend();
preds = []
for batch_i, (data, target) in enumerate(test_loader):
    data = data.cuda()
    output = model(data)
    pr = output[:,1].detach().cpu().numpy()
    for i in pr:
        preds.append(i)
test_preds = pd.DataFrame({'imgs': test_set.image_files_list, 'preds': preds})
test_preds['imgs'] = test_preds['imgs'].apply(lambda x: x.split('.')[0])
sub = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')
sub = pd.merge(sub, test_preds, left_on='id', right_on='imgs')
sub = sub[['id', 'preds']]
sub.columns = ['id', 'label']
sub.head()
sub.to_csv('submission.csv', index=False)