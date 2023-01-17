# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train_labels.csv")
df.head()
from PIL import Image
import matplotlib.pyplot as plt

example_image = Image.open("../input/train/" + df['id'][0] + ".tif")
example_image_numpy = np.array(example_image.getdata())

#Finding the dimensions of the image
print(example_image_numpy.shape)
print(np.sqrt(example_image_numpy.shape[0]))
example_image_numpy = np.array(example_image.getdata()).reshape(96, 96, 3)

plt.imshow(example_image_numpy)
plt.show()
example_images_numpy = []
example_labels = []
image_num = np.random.choice(df.shape[0] - 21)

# We will view around 20 consecutive examples

for i in range(image_num, image_num + 20):
    example_image = Image.open("../input/train/" + df['id'][i] + ".tif")
    example_images_numpy.append(np.array(example_image.getdata()).reshape(96, 96, 3))
    example_labels.append(str(df['label'][i]))

# Setting up matplotlib
fig, ax = plt.subplots(4, 5, figsize = (25, 20))

k = 0
# Setting up the axes objects
for i in range(4):
    for j in range(5):
        ax[i, j].imshow(example_images_numpy[k])
        ax[i, j].axis('off')
        ax[i, j].set_title(example_labels[k])
        k += 1
# Viewing the plots
plt.show()
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch import optim, save
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable
print(torch.__version__)
class HS_Dataset(Dataset):
    
    def __init__(self, csv_file, root_dir, transform = None):
        self.df = pd.read_csv(csv_file)
        self.dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        """This function should return the ith example from the training set.
        The example should be returned in the form of a dictionary: 
        {'image': image_data, 'label': label_data}"""
        
        file = df['id'][i]
        
        label = np.array(df['label'][i])
        if label == 0:
            label == 0.0
        else:
            label == 1.0
            
        """Reshape needed to make the output of shape [1]"""
        label = label.reshape((1))
                
        image = Image.open("../input/train/" + file + ".tif")
        image = np.array(image.getdata()).reshape(96, 96, 3)
        
        sample = {'image': image, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
        
        
class ToTensor(object):
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        """This transposition is very important as PyTorch take in the image data in the current shape:
        Number of Channels, Height, Width; So the third axis(channels) in the original image has to 
        be made the first axis."""
        image = image.transpose(2, 0, 1)        
        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)
        
        label = torch.from_numpy(label)
        label = label.type(torch.FloatTensor)
        """The optimizer takes in FloatTensor type data. Hence the data has to be converted from any other format
        to FloatTensor type"""
        return {'image': image, 'label': label}
from torch.utils.data import random_split

hs_dataset = HS_Dataset("../input/train_labels.csv", "../input/train/", transform = transforms.Compose([ToTensor()]))

train_size = int(0.995 * len(hs_dataset))
val_size = int((len(hs_dataset) - train_size) / 8)
test_size = int(val_size * 7 / 8)
test_size += len(hs_dataset) - train_size - val_size - test_size

print("train size: ", train_size)
print("val size: ", val_size)
print("test size: ", test_size)

train_data, val_data, test_data = random_split(hs_dataset, [train_size, val_size, test_size])
print("training")
for i in range(len(train_data)):
    sample = train_data[i]
    print(i, sample['image'].size(), sample['label'].size())
    if i == 5:
        break

print("validation")
for i in range(len(val_data)):
    sample = val_data[i]
    print(i, sample['image'].size(), sample['label'].size())
    if i == 5:
        break
        
print("testing")
for i in range(len(test_data)):
    sample = test_data[i]
    print(i, sample['image'].size(), sample['label'].size())
    if i == 5:
        break
train_loader = DataLoader(dataset = train_data, batch_size = 128, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset= val_data, batch_size = val_size, num_workers=0)
test_loader = DataLoader(dataset= test_data, batch_size = 128, num_workers=0)
print(len(train_loader))
print(len(val_loader))
print(len(test_loader))
class SkipConvNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(SkipConvNet, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(num_channels, 8, kernel_size=3, padding = 1),
                                  nn.BatchNorm2d(8),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.skip1_1 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, padding = 1),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU(),
                                    nn.Conv2d(8, 8, kernel_size=3, padding = 1))
        
        self.skip1_2 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, padding = 1),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU(),
                                    nn.Conv2d(8, 8, kernel_size=3, padding = 1),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU(),
                                    nn.Conv2d(8, 8, kernel_size=3, padding = 1))
        
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, padding = 1),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.skip2_1 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding = 1),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.Conv2d(16, 16, kernel_size=3, padding = 1))
        
        self.skip2_2 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding = 1),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.Conv2d(16, 16, kernel_size=3, padding = 1),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.Conv2d(16, 16, kernel_size=3, padding = 1))
        
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.skip3_1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding = 1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 32, kernel_size=3, padding = 1))
        
        self.skip3_2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding = 1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 32, kernel_size=3, padding = 1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 32, kernel_size=3, padding = 1))
        
        self.conv4 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding = 1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.skip4_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding = 1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, padding = 1))
        
        self.skip4_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding = 1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, padding = 1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, padding = 1))
        
        self.conv5 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.skip5_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding = 1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, kernel_size=3, padding = 1))
        
        self.skip5_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding = 1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, kernel_size=3, padding = 1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, kernel_size=3, padding = 1))
        
         
        self.ff1 = nn.Linear(3 * 3 * 128, 128)
        self.ff2 = nn.Linear(128, 32)
        
        self.output = nn.Linear(32, num_classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = out + self.skip1_1(out) + self.skip1_2(out)
        out = self.conv2(out)
        out = out + self.skip2_1(out) + self.skip2_2(out)
        out = self.conv3(out)
        out = out + self.skip3_1(out) + self.skip3_2(out)
        out = self.conv4(out)
        out = out + self.skip4_1(out) + self.skip4_2(out)
        out = self.conv5(out)
        out = out + self.skip5_1(out) + self.skip5_2(out)
        out = out.reshape(-1, 3 * 3 * 128)
        out = self.ff1(out)
        out = self.ff2(out)
        out = self.output(out)
        return out
model = SkipConvNet(3, 1)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.BCEWithLogitsLoss()
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def accuracy_mini_batch(predicted, true, i, acc, tpr, tnr):
    
    predicted = predicted.cpu()
    true = true.cpu()
    
    predicted = (sigmoid(predicted.data.numpy()) > 0.5)
    true = true.data.numpy()
    
    accuracy = np.sum(predicted == true) / true.shape[0]
    true_positive_rate = np.sum((predicted == 1) * (true == 1)) / np.sum(true == 1)
    true_negative_rate = np.sum((predicted == 0) * (true == 0)) / np.sum(true == 0)
    
    acc = acc * (i) / (i + 1)  + accuracy / (i + 1)
    tpr = tpr * (i) / (i + 1)  + true_positive_rate / (i + 1)
    tnr = tnr * (i) / (i + 1) + true_negative_rate / (i + 1)
    
    return acc, tpr, tnr


def accuracy(predicted, true):
    predicted = predicted.cpu()
    true = true.cpu()
    
    predicted = (sigmoid(predicted.data.numpy()) > 0.5)
    true = true.data.numpy()
    
    accuracy = np.sum(predicted == true) / true.shape[0]
    true_positive_rate = np.sum((predicted == 1) * (true == 1)) / np.sum(true == 1)
    true_negative_rate = np.sum((predicted == 0) * (true == 0)) / np.sum(true == 0)

    return accuracy, true_positive_rate, true_negative_rate
import time
import matplotlib.pyplot as plt

epochs = 10

accuracy_array = []
tpr_array = []
tnr_array = []
loss_array = []

val_loss_array = []
val_acc_array = []
val_tpr_array = []
val_tnr_array = []

use_cuda = torch.cuda.is_available()
device = "cuda:0"

for epoch in range(epochs):
    start_time = time.time() 
    
    loss_temp = []
    
    acc, tpr, tnr = 0., 0., 0.
    
    for mini_batch_num, data in enumerate(train_loader):
        images, labels = data['image'], data['label']
        images, labels = images.to(device), labels.to(device)
        
        preds = model(images)
        
        loss = criterion(preds, labels)
        acc, tpr, tnr = accuracy_mini_batch(preds, labels, i, acc, tpr, tnr)
        
        optimizer.zero_grad()
        loss.backward()
        loss_temp.append(loss.item())
        
        optimizer.step()
        if (mini_batch_num) % 4 == 0:
            print ('Epoch {}/{}; Iter {}/{}; Loss: {:.4f}; Acc: {:.3f}; True Pos: {:.3f}; True Neg: {:.3f}'
                   .format(epoch+1, epochs, mini_batch_num + 1, len(train_loader), loss.item(), acc, tpr, tnr), end = "\r", flush = True)
    
    end_time = time.time()
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, labels = data['image'], data['label']
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss_test = criterion(preds, labels)
            t_acc, t_tpr, t_tnr = accuracy(preds, labels)
    
    val_loss_array.append(loss_test)
    val_acc_array.append(t_acc)
    val_tpr_array.append(t_tpr)
    val_tnr_array.append(t_tnr)
    
    print ('Epoch {}/{}; Loss: {:.4f}; Train Acc: {:.3f}; Train TPR: {:.3f}; Train TNR: {:.3f}; Epoch Time: {} mins; \nTest Loss: {:.4f}; Test Acc: {:.3f}; Test TPR: {:.3f}; Test TNR: {:.3f}\n'
           .format(epoch+1, epochs, loss.item(),acc, tpr, tnr, round((end_time - start_time)/ 60., 2), loss_test, t_acc, t_tpr, t_tnr))
    
    loss_array.append(np.mean(np.array(loss_temp)))
    accuracy_array.append(acc)
    tpr_array.append(tpr)
    tnr_array.append(tnr)
plt.plot(loss_array, color="red")
plt.plot(accuracy_array, color="blue")
plt.plot(tpr_array, color="green")
plt.plot(tnr_array, color="orange")
plt.show()
plt.plot(val_loss_array, color="red")
plt.plot(val_acc_array, color="blue")
plt.plot(val_tpr_array, color="green")
plt.plot(val_tnr_array, color="orange")
plt.show()
