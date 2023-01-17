import numpy as np # to handle matrix and data operation
import pandas as pd # to read csv and handle dataframe

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import random

use_cuda=True
class DatasetMNIST(Dataset):
    
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        label = self.data.iloc[index, 0]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
transform=transforms.Compose([
                            transforms.ToPILImage(),
                           # Add random transformations to the image.
                           transforms.RandomAffine(
                               degrees=5, translate=(0.1, 0.2), scale=(0.55, 1.2),
                               shear=(-20, 10, -20, 10)),

                           transforms.ToTensor()
                       ])
train_dataset = DatasetMNIST('/kaggle/input/Kannada-MNIST/train.csv', transform=transform)
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
fig=plt.figure(figsize=(28, 28))
columns = 12
rows = 1
for i in range(1, columns*rows +1):
    r=random.randint(2,int(len(train_dataset)/columns*rows))
    img, lab = train_dataset.__getitem__(i*r)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img.squeeze())
    plt.title(lab)
plt.show()
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(15488, 10)
#         self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output
net = Net()
model = net.cuda()
# train

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)#, momentum=0.9)
error = nn.CrossEntropyLoss()
EPOCHS = 10

model.train()
for epoch in range(EPOCHS):
    correct = 0
    running_loss=0
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        var_X_batch = Variable(X_batch).float()
        var_y_batch = Variable(y_batch)
        var_X_batch = var_X_batch.cuda()
        var_y_batch = var_y_batch.cuda()
        optimizer.zero_grad()
        output = model(var_X_batch)
        loss = error(output, var_y_batch)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()

        # Total correct predictions
        predicted = torch.max(output.data, 1)[1] 
        correct += (predicted == var_y_batch).sum()
       
    print('Epoch : {} \t\tLoss: {:.5f}\t Accuracy:{:.2f}%'.format(
                epoch, 
                100.*batch_idx / len(train_loader),
                loss.item(), 
                float(correct*100) / float(batch_size*(batch_idx+1))))
