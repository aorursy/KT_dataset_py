!ls -1 ./target/valid | wc -l
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

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

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

from collections import OrderedDict
import cv2
full_train_df = pd.read_csv("../input/humpback-whale-identification/train.csv")
full_train_df.head()
print(f"There are {len(os.listdir('../input/humpback-whale-identification/train'))} images in train dataset with {full_train_df.Id.nunique()} unique classes.")
print(f"There are {len(os.listdir('../input/humpback-whale-identification/test'))} images in test dataset.")
fig = plt.figure(figsize=(25, 4))
train_imgs = os.listdir("../input/humpback-whale-identification/train")
for idx, img in enumerate(np.random.choice(train_imgs, 20)):
    ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
    im = Image.open("../input/humpback-whale-identification/train/" + img)
    plt.imshow(im)
    lab = full_train_df.loc[full_train_df.Image == img, 'Id'].values[0]
    ax.set_title(f'Label: {lab}')
full_train_df.Id.value_counts().sort_values(ascending=False).head()
import copy
new_whale_df = full_train_df.query("Id == 'new_whale'")
train_df = full_train_df.query("Id != 'new_whale'")
print(new_whale_df.shape)
print(train_df.shape)
if not os.path.exists('./target'):
    os.system("mkdir ./target")
if not os.path.exists('./target/train'):
    os.system("mkdir ./target/train")
for image_name in train_df.Image.values:
    src_path = os.path.join('../input/humpback-whale-identification/train', image_name)
    dist_path = os.path.join('./target/train')
    os.system("cp " + src_path + " " + dist_path)
for i in range(1, 4):
    print(f'There are {train_df.Id.value_counts()[train_df.Id.value_counts().values==i].shape[0]} classes with {i} samples in train data.')
plt.title('Distribution of classes excluding new_whale');
train_df.Id.value_counts()[1:].plot(kind='hist');
np.array(im).shape
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
class WhaleDataset(Dataset):
    def __init__(self, datafolder, datatype='train', df=None, transform = transforms.Compose([transforms.ToTensor()]), y=None):
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
            label = np.zeros((5004,))
            
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        if self.datatype == 'train':
            return image, label
        elif self.datatype == 'test':
            # so that the images will be in a correct order
            return image, label, self.image_files_list[idx]
y, le_full = prepare_labels(train_df['Id'])
data_transforms = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
train_dataset = WhaleDataset(datafolder='./target/train/', datatype='train', df=train_df, transform=data_transforms, y=y)

batch_size = 32
num_workers = 2

train_sampler = SubsetRandomSampler(list(range(len(os.listdir('./target/train')))))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)        
        self.pool2 = nn.AvgPool2d(3, 3)
        
        self.fc1 = nn.Linear(64 * 4 * 4 * 16, 1024)
        self.fc2 = nn.Linear(1024, 5004)

        self.dropout = nn.Dropout(0.5)        

    def forward(self, x):
        x = self.pool(F.relu(self.conv2_bn(self.conv1(x))))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
model_conv = Net()
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model_conv.parameters(), lr=0.01)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
model_conv.cuda()
n_epochs = 7
loss_list = []
for epoch in range(1, n_epochs+1):
    print(time.ctime(), 'Epoch:', epoch)

    train_loss = []
    exp_lr_scheduler.step()

    for batch_i, (data, target) in  enumerate(train_loader):
        #print(batch_i)
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model_conv(data)
        loss = criterion(output, target.float())
        train_loss.append(loss.item())

        loss.backward()
        optimizer.step()
        
    loss_list.append(np.mean(train_loss))
    print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}')
plt.plot(range(n_epochs), loss_list, 'r-', label='train_loss')
plt.legend()
torch.save(model_conv, './without_newwhale_model')
len_valid_nw = new_whale_df.shape[0]
valid_new_whale_df = new_whale_df.head(int(len_valid_nw * 0.2))
valid_df = pd.concat([train_df, valid_new_whale_df]).reset_index(drop=True)
valid_df.head()
del train_df
if not os.path.exists('./target/valid'):
    os.system("mkdir ./target/valid")
!rm -f ./target/valid/*
!rm -f ./target/train/*
valid_df.shape
for image_name in valid_df.Image.values:
    src_path = os.path.join('../input/humpback-whale-identification/train', image_name)
    dist_path = os.path.join('./target/valid')
    os.system("cp " + src_path + " " + dist_path)
!ls -1 ./target/valid | wc -l
y, le_full = prepare_labels(valid_df['Id'])
data_transforms_valid = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

valid_set = WhaleDataset(datafolder='./target/valid', datatype='test', transform=data_transforms_valid)

batch_size = 32
num_workers = 2

valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=32, num_workers=num_workers, pin_memory=True)
model_conv = torch.load('./without_newwhale_model')
model_conv.eval()
for (data, target, name) in valid_loader:
    data = data.cuda()
    output = model_conv(data)
    output = output.cpu().detach().numpy()
    for i, (e, n) in enumerate(list(zip(output, name))):
        e_str = [str(s) for s in e]
        valid_df.loc[valid_df['Image'] == n, 'Predict'] = ' '.join(e_str)

valid_df.head()
preds = valid_df['Predict']
ids = valid_df['Id']
def map5(X, y, th):
    score = 0
    for i in range(X.shape[0]):
        str_X = X[i].split(' ')
        result = [float(s) for s in str_X]
        result.insert(0, float(th))
        result = np.array(result)
        pred = le_full.inverse_transform(result.argsort()[-5:][::-1])
        for j in range(pred.shape[0]):
            if pred[j] == y[i]:
                score += (5 - j)/5
                break
    return float(score/X.shape[0])
best_th = 0
best_score = 0
for th in np.arange(-10, 0, 1):
    score = map5(preds, ids, th)
    if score > float(best_score):
        best_score = score
        best_th = th
    print("Threshold = {:.3f}, MAP5 = {:.3f}".format(th,score))
print("Best Threshold = {:.3f}, Best MAP5 = {:.3f}".format(best_th,best_score))
data_transforms_test = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_set = WhaleDataset(datafolder='../input/humpback-whale-identification/test/', datatype='test', transform=data_transforms_test)

batch_size = 32
num_workers = 2

test_sampler = SubsetRandomSampler(list(range(len(os.listdir('../input/humpback-whale-identification/test')))))
# less size for test loader.
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, num_workers=num_workers)
sub = pd.read_csv('../input/humpback-whale-identification/sample_submission.csv')

model_conv.eval()
for index, (data, target, name) in enumerate(test_loader):
    data = data.cuda()
    output = model_conv(data)
    output = output.cpu().detach().numpy()
    for i, (e, n) in enumerate(list(zip(output, name))):
        e = np.insert(e, 0, float(best_th))
        sub.loc[sub['Image'] == n, 'Id'] = ' '.join(le_full.inverse_transform(e.argsort()[-5:][::-1]))
        
sub.to_csv('submission_without_new_whale.csv', index=False)
!rm -rf ./target
