# libraries

import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

import torch

from torch.utils.data import TensorDataset, DataLoader,Dataset

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

import torch.optim as optim

from torch.optim import lr_scheduler

import time 

import tqdm

from PIL import Image

train_on_gpu = True

from torch.utils.data.sampler import SubsetRandomSampler

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR



import cv2

import albumentations

from albumentations import pytorch as AT

import glob

from sklearn.preprocessing import OneHotEncoder
class DigitDataset(Dataset):

    def __init__(self, datafolder='/kaggle/input/handwritten-digits',

                 transform = transforms.Compose([transforms.ToTensor()])):

        self.datafolder = datafolder

        self.transform = transform

        self.image_files_list = []

        self.labels = []

        self._load_images()

    

    def _load_images(self):

        digit_folders = os.listdir(self.datafolder)

        for folder in digit_folders:

            for i, pic in enumerate(glob.glob(os.path.join(self.datafolder, folder, '*.jpg'))):



                img = Image.open(pic).convert('RGB')

                bbox = Image.eval(img, lambda px: 255-px).getbbox()

                if img.crop(bbox) not in self.image_files_list:

                    self.image_files_list.append(img.crop(bbox))

                    if folder == 'other1':

                        self.image_files_list.append(img.crop(bbox))

                        self.image_files_list.append(img.crop(bbox))



                    if folder != 'other1':

                        # print(pic)

                        if '__' in pic:

                            self.labels.append(int(pic.split('/')[-1].split('__')[0][-1]))

                        else:

                            self.labels.append(int(pic.split('/')[-1].split('_')[1]))

                    else:

                        for _ in range(3):

                            self.labels.append(10)

    

    def __len__(self):

        return len(self.image_files_list)



    def __getitem__(self, idx):

        image = self.image_files_list[idx]

        image = self.transform(image)

        label = self.labels[idx]

        weight = self.weights[idx]



        return image, label, weight
train_transforms = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.RandomHorizontalFlip(p=0.2),

    transforms.RandomRotation((-15, 15)),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])



test_transforms = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])
dataset = DigitDataset(datafolder='/kaggle/input/handwritten-digits', transform=train_transforms)
len(dataset)
fig = plt.figure(figsize=(25, 4))

# display 20 images

for idx, img_id in enumerate(np.random.randint(0, len(dataset), 20)):

    ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])

    plt.imshow(dataset.image_files_list[img_id])

    lab = dataset.labels[img_id]

    ax.set_title(f'Label: {lab}')
onehot_encoder = OneHotEncoder(sparse=False)

onehot_encoder.fit(np.arange(10).reshape(-1, 1))

ohe_labels = onehot_encoder.transform(np.array(dataset.labels).reshape(-1, 1))

dataset.labels = ohe_labels
weights = []

for i in np.unique(dataset.labels.argmax(1), return_counts=True)[1]:

    weights.extend([len(dataset.labels) / i] * i)

    

dataset.weights = weights
tr, val = train_test_split(range(len(dataset.labels)),

                           stratify=dataset.labels, test_size=0.1)



train_sampler = SubsetRandomSampler(list(tr))

valid_sampler = SubsetRandomSampler(list(val))

batch_size = 128

num_workers = 0

# prepare data loaders (combine dataset and sampler)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,

                                           sampler=train_sampler, num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,

                                           sampler=valid_sampler, num_workers=num_workers)
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3)

        self.fc1 = nn.Linear(576 * 2, 512)

        self.fc2 = nn.Linear(512, 10)

        self.dropout = nn.Dropout(0.1)



    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 576 * 2)

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.fc2(x)

        return x

model_conv = Net()

model_conv.cuda()

criterion = nn.BCEWithLogitsLoss()



optimizer = optim.SGD(model_conv.parameters(), lr=0.1, momentum=0.85)

model_scheduler = CosineAnnealingLR(optimizer, T_max=5)
valid_loss_min = np.Inf

valid_loss_hist = []

train_loss_hist = []

best_epoch = 0

patience = 15

# current number of epochs, where validation loss didn't increase

p = 0

# whether training should be stopped

stop = False



# number of epochs to train the model

n_epochs = 100

train_accuracy = []

valid_accuracy = []

for epoch in range(1, n_epochs+1):

    print(time.ctime(), 'Epoch:', epoch)



    train_loss = []

    train_acc = []



    for batch_i, (data, target, weight) in enumerate(train_loader):



        data, target, weight = data.cuda(), target.cuda(), weight.cuda()



        optimizer.zero_grad()

        output = model_conv(data)

        criterion.weight = weight.view(-1, 1).double()

        loss = criterion(output.double(), target.double())

        train_loss.append(loss.item())

        

        a = target.data.cpu().numpy()

        b = output[:,-1].detach().cpu().numpy()

        train_acc.append(sum(np.argmax(a, axis=1) == output.argmax(1).cpu().numpy()) / len(a))

        # train_auc.append(roc_auc_score(a, b))



        loss.backward()

        optimizer.step()

    

    model_conv.eval()

    val_loss = []

    val_acc = []

    for batch_i, (data, target, weight) in enumerate(valid_loader):

        data, target, weight = data.cuda(), target.cuda(), weight.cuda()

        output = model_conv(data)

        criterion.weight = weight.view(-1, 1).double()

        loss = criterion(output.double(), target.double())



        val_loss.append(loss.item()) 

        a = target.data.cpu().numpy()

        b = output[:,-1].detach().cpu().numpy()

        val_acc.append(sum(np.argmax(a, axis=1) == output.argmax(1).cpu().numpy()) / len(a))

        # val_auc.append(roc_auc_score(a, b))



    print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}, valid loss: {np.mean(val_loss):.4f}, train acc: {np.mean(train_acc):.4f}, valid acc: {np.mean(val_acc):.4f}')

    train_accuracy.append(np.mean(train_acc))

    valid_accuracy.append(np.mean(val_acc))

    valid_loss = np.mean(val_loss)

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min,

        valid_loss))

        # torch.save(model_conv.state_dict(), 'model2____1.pt')

        valid_loss_min = valid_loss

        p = 0

        best_epoch = epoch

    valid_loss_hist.append(valid_loss)

    train_loss_hist.append(np.mean(train_loss))



    # check if validation loss didn't improve

    if valid_loss > valid_loss_min:

        p += 1

        print(f'{p} epochs of increasing val loss')

        if p > patience:

            print('Stopping training')

            stop = True

            break        

    

    model_scheduler.step(epoch)

    

    if stop:

        break

        

print(f'Best train_accuracy: {max(train_accuracy)* 100:.4f}%. Best valid_accuracy: {max(valid_accuracy)* 100:.4f}%. Loss: {valid_loss_min:.4f}')
targets = []

all_output = []

model_conv.eval()

for batch_i, (data, target, weight) in enumerate(valid_loader):



    data, target = data.cuda(), target.cuda()



    optimizer.zero_grad()

    output = model_conv(data)

    targets.extend(target.argmax(1).cpu().numpy())

    all_output.extend(output.argmax(1).cpu().numpy())
from sklearn import metrics

# Show confusion table

plt.rcParams['figure.figsize'] = (8.0, 8.0)

conf_matrix = metrics.confusion_matrix(targets, all_output, labels=None)  # Get confustion matrix

# Plot the confusion table

class_names = ['${:d}$'.format(x) for x in range(0, 11)]  # Digit class names

fig = plt.figure()

ax = fig.add_subplot(111)

# Show class labels on each axis

ax.xaxis.tick_top()

major_ticks = range(0,11)

minor_ticks = [x + 0.5 for x in range(0, 11)]

ax.xaxis.set_ticks(major_ticks, minor=False)

ax.yaxis.set_ticks(major_ticks, minor=False)

ax.xaxis.set_ticks(minor_ticks, minor=True)

ax.yaxis.set_ticks(minor_ticks, minor=True)

ax.xaxis.set_ticklabels(class_names, minor=False, fontsize=15)

ax.yaxis.set_ticklabels(class_names, minor=False, fontsize=15)

# Set plot labels

ax.yaxis.set_label_position("right")

ax.set_xlabel('Predicted label')

ax.set_ylabel('True label')

fig.suptitle('Confusion table', y=1.03, fontsize=15)

# Show a grid to seperate digits

ax.grid(b=True, which=u'minor')

# Color each grid cell according to the number classes predicted

ax.imshow(conf_matrix, interpolation='nearest', cmap='binary')

# Show the number of samples in each cell

for x in range(10):

    for y in range(10):

        color = 'w' if x == y else 'k'

        ax.text(x, y, conf_matrix[y,x], ha="center", va="center", color=color)       

plt.show()
stats_dict = {'train_acc': train_accuracy, 'valid_acc': valid_accuracy,

              'train_loss': train_loss_hist, 'valid_loss': valid_loss_hist, 'best_epoch': best_epoch}

plt.figure(figsize=(12, 8))

plt.plot(stats_dict['train_acc'], label='train_accuracy');

plt.plot(stats_dict['valid_acc'], label='valid_accuracy');

plt.title('Accuracy while training');

plt.axvline(x=stats_dict['best_epoch'], color='red', label='best epoch')

plt.legend();

plt.xlabel('Epoch');

plt.ylabel('Accuracy');
plt.figure(figsize=(12, 8))

plt.plot(stats_dict['train_loss'], label='train_loss');

plt.plot(stats_dict['valid_loss'], label='valid_loss');

plt.title('Loss while training');

plt.axvline(x=stats_dict['best_epoch'], color='red', label='best epoch')

plt.legend();

plt.xlabel('Epoch');

plt.ylabel('Loss');