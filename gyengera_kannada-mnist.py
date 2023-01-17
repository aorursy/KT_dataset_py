# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print ("Reading data")
train_set = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")
val_set = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")
test_set = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

train_set = pd.concat([train_set, val_set], ignore_index=True)

train_images = train_set.iloc[:, 1:]
train_labels = train_set.iloc[:, 0]
val_images = val_set.iloc[:, 1:]
val_labels = val_set.iloc[:, 0]
test_images = test_set.iloc[:, 1:]
train_transform = transforms.Compose(([
        transforms.ToPILImage(),
        transforms.RandomCrop(28),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
        transforms.ToTensor(), # divides by 255
        ]))

test_transform = transforms.Compose(([
        transforms.ToPILImage(),
        transforms.ToTensor(), # divides by 255
        ]))
class KannadaDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
                    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        #get item and convert it to numpy array
        data = self.X.iloc[i, :]
        data = np.array(data).astype(np.uint8).reshape(28, 28, 1) 
        
        # perform transforms if there are any
        if self.transforms:
            data = self.transforms(data)
        
        if self.y is not None: # train/val
            return (data, self.y[i])
        else: # test
            return data
        

print ("Dataset class created")
print ("Defining network architecture")
class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.conv1_5 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.conv1_5_bn = nn.BatchNorm2d(num_features=64)
        self.conv1_3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv1_3_bn = nn.BatchNorm2d(num_features=64)

        self.conv2_5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2)
        self.conv2_5_bn = nn.BatchNorm2d(num_features=128)
        self.conv2_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(num_features=128)

        self.conv3_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2)
        self.conv3_5_bn = nn.BatchNorm2d(num_features=256)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(num_features=256)

        self.fc1 = nn.Linear(in_features=512*6*6, out_features=512)
        self.fc1_bn = nn.BatchNorm1d(num_features=512)

        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc2_bn = nn.BatchNorm1d(num_features=256)
        
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc3_bn = nn.BatchNorm1d(num_features=128)

        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc4_bn = nn.BatchNorm1d(num_features=64)

        self.out = nn.Linear(in_features=64, out_features=10)


    def forward(self, x):
        x_5 = F.relu(self.conv1_5_bn(self.conv1_5(x))) #(64, 28, 28)
        x_3 = F.relu(self.conv1_3_bn(self.conv1_3(x))) #(64, 28, 28)
        x = F.max_pool2d(torch.cat((x_5, x_3), dim=1), kernel_size=2, stride=2) #(128, 14, 14)

        x_5 = F.relu(self.conv2_5_bn(self.conv2_5(x))) #(128, 14, 14)
        x_3 = F.relu(self.conv2_3_bn(self.conv2_3(x))) #(128, 14, 14)
        x = F.max_pool2d(torch.cat((x_5, x_3), dim=1), kernel_size=2, stride=2) #(256, 7, 7)

        x_5 = F.relu(self.conv3_5_bn(self.conv3_5(x))) #(256, 7, 7)
        x_3 = F.relu(self.conv3_3_bn(self.conv3_3(x))) #(256, 7, 7)
        x = F.max_pool2d(torch.cat((x_5, x_3), dim=1), kernel_size=2, stride=1) #(512, 6, 6)

        x = F.relu(self.fc1_bn(self.fc1(x.view(-1, 6*6*512)))) #(512)
        x = F.relu(self.fc2_bn(self.fc2(x))) #(256)
        x = F.relu(self.fc3_bn(self.fc3(x))) #(128)
        x = F.relu(self.fc4_bn(self.fc4(x))) #(64)
        x = self.out(x) #(10)

        return x
net = network().to(device)
lr = 0.001
batch_size = 100
max_epochs = 15
train_set = KannadaDataset(train_images, train_labels, train_transform)
val_set = KannadaDataset(val_images, val_labels, test_transform)
train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_data = DataLoader(val_set, batch_size=batch_size, shuffle=False)

print ("Begin training")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, 4, gamma=0.1)
for epoch in range(max_epochs):
    epoch_loss = 0.
    epoch_accuracy = 0.

    net.train()
    for i, (images, labels) in enumerate(train_data):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, max_epochs, i+1, len(train_set)//batch_size, loss.item()))
    
#     net.eval()
#     with torch.no_grad():
#         val_accuracy = 0.
#         for images, labels in val_data:
#             images = images.to(device)
#             labels = labels.to(device)

#             outputs = net(images)
#             _, predicted = torch.max(outputs, 1)
#             val_accuracy += (predicted==labels).sum().item()
#     print('Validation accuracy is: {} %'.format(100 * val_accuracy / len(val_set)))
    scheduler.step()
test_set = KannadaDataset(images=test_images, transforms=test_transform)
test_data = DataLoader(test_set, batch_size=1, shuffle=False)

net.eval()
predictions = torch.LongTensor().to(device)
with torch.no_grad():
    for images in test_data:
        images = images.to(device)

        outputs = net(images)
        predictions = torch.cat((predictions, outputs.argmax(dim=1)), dim=0)

submission = pd.read_csv("/kaggle/input/Kannada-MNIST/sample_submission.csv")
submission['label'] = predictions.cpu().numpy()
submission.to_csv("/kaggle/working/submission.csv", index=False)
print ("Output written to csv file!")