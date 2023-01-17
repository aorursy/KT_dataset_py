import os

from os import path

import shutil

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import Dataset

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets, transforms, models

import matplotlib

import matplotlib.pyplot as plt

from PIL import Image
PATH = '../input/numta/'

os.listdir(PATH)
def showRawTrainingSamples(csv_filename):

  df = pd.read_csv(PATH + csv_filename)

  print(csv_filename)

  print(df.columns)

  return df
a_csv = showRawTrainingSamples('training-a.csv')

c_csv = showRawTrainingSamples('training-c.csv')

d_csv = showRawTrainingSamples('training-d.csv')
def dropColumns(csv_file):

  csv_file = csv_file[['filename', 'digit']]

  print(csv_file)

  print(csv_file.iloc[:5, :])   #First 5 Rows of the CSV File

  print("=============================")

  return csv_file
a_csv = dropColumns(a_csv)

c_csv = dropColumns(c_csv)

d_csv = dropColumns(d_csv)
total_csv = [a_csv, c_csv, d_csv]

merged_csv = pd.concat(total_csv)

print(len(merged_csv))
TRAIN_PATH = 'train'

os.mkdir(TRAIN_PATH)
def processImages(folder_name):

  src = PATH + folder_name + '/'

  dir_folders = os.listdir(src)

  for dir_name in dir_folders:

    file_name = os.path.join(src, dir_name)

    if os.path.isfile(file_name):

      shutil.copy(file_name, TRAIN_PATH)  
processImages('training-a')

print('A Done')

processImages('training-c')

print('C Done')

processImages('training-d')

print('D Done')
class Dataset(Dataset):

    def __init__(self, df, root, transform=None):

        self.data = df

        self.root = root

        self.transform = transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        item = self.data.iloc[index]

        

        path = self.root + "/" + item[0]

        image = Image.open(path).convert('L')

        label = item[1]

        

        if self.transform is not None:

            image = self.transform(image)

            

        return image, label
mean = [0.5,]

std = [0.5, ]



train_transform = transforms.Compose([

    transforms.Resize(28),

    transforms.ToTensor(),

    transforms.Normalize(mean, std)

])



test_transform = transforms.Compose([

        transforms.Resize(28),

        transforms.ToTensor(),

        transforms.Normalize(mean, std)

])



train_data  = Dataset(merged_csv, TRAIN_PATH, train_transform)

test_data = Dataset(merged_csv, TRAIN_PATH, test_transform)



print("Trainig Samples: ",len(train_data))
#batch size

batch_size = 32



# split data 20% for testing

test_size = 0.2



# obtain training indices that will be used for validation

num_train = len(train_data)



# mix data

# index of num of train

indices = list(range(num_train))

# random the index

np.random.shuffle(indices)

split = int(np.floor(test_size * num_train))

# divied into two part

train_idx, test_idx = indices[split:], indices[:split]



# define the sampler

train_sampler = SubsetRandomSampler(train_idx)

test_sampler = SubsetRandomSampler(test_idx)



# prepare loaders

train_loader = torch.utils.data.DataLoader(

    train_data, batch_size=batch_size,

    sampler=train_sampler)



test_loader = torch.utils.data.DataLoader(

    test_data, batch_size=batch_size,

    sampler=test_sampler)



print("Train dataloader:{}".format(len(train_loader)))

print("Test dataloader:{}".format(len(test_loader)))
input_dim = 28 * 28

output_dim = 10

class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(input_dim, 512)

        self.fc2 = nn.Linear(512, 256)

        self.fc3 = nn.Linear(256, 128)

        self.fc4 = nn.Linear(128, output_dim)

        

    

    def forward(self, x):

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = F.log_softmax(self.fc4(x), dim=1)

        return x
USE_GPU = True

device = 'cpu'

if USE_GPU and torch.cuda.is_available():

    device = 'cuda'

model = Net()

model.to(device)
learning_rate = 1e-4

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr = learning_rate)
epochs = 200

test_loss_min = np.Inf

train_loss_data, test_loss_data = [], []

iteration = 0



for e in range(epochs):

    running_loss = 0

    train_loss = 0.0

    test_loss = 0.0

    total = 0

    correct = 0

    print("Epoch:", e+1)

    

    for images, labels in train_loader:

        images, labels = images.view(images.shape[0], -1).to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * images.size(0)





    for images, labels in test_loader:

        images, labels = images.view(images.shape[0], -1).to(device), labels.to(device)

        logits = model(images)

        loss = criterion(logits, labels)

        test_loss += loss.item() * images.size(0)

        proba = torch.exp(logits)

        top_p, top_class = proba.topk(1, dim=1)

        equals = top_class == labels.view(*top_class.shape)



        _, predicted = torch.max(logits.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()





    train_loss = train_loss / len(train_loader.dataset)

    test_loss = test_loss / len(test_loader.dataset)



    train_loss_data.append(train_loss * 100)

    test_loss_data.append(test_loss * 100)



    accuracy = (correct / total) * 100



    print("\tTrain loss:{:.6f}..".format(train_loss),

            "\tValid Loss:{:.6f}..".format(test_loss),

            "\tAccuracy: {:.4f}".format(accuracy))

    

    if test_loss <= test_loss_min:

        print('\tValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(test_loss_min,test_loss))

        torch.save(model.state_dict(), 'model_final.pt')

        test_loss_min = test_loss



# Check out the notebook below for better documentation

# https://www.kaggle.com/iamsdt/bn-digit-with-pytorch (Pytorch CNN)