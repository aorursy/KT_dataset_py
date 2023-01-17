import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
print(torch.__version__)
torch.cuda.is_available()
batch_size = 64
epochs = 20
class FashionMNIST(Dataset):
    
    def __init__(self, X, y=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X.index)
    
    def __getitem__(self, index):
        image = self.X.iloc[index, :].values.astype(np.uint8).reshape((28, 28, 1)) # (H, W, C)
        
        if self.transform is not None:
            image = self.transform(image)
    
        if self.y is not None: # training 
            return image, self.y.iloc[index]
        else: # test 
            return image
df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')

X_train, X_valid, y_train, y_valid = train_test_split(df_train.iloc[:, 1:], df_train['label'], test_size=1/6, random_state=42)
X_test = df_test

print('train data shape  : {}\t dtype {}'.format(X_train.shape, type(X_train)))
print('train label shape : {}\t\t dtype {}'.format(y_train.shape, type(y_train)))
print('valid data shape  : {}\t dtype {}'.format(X_valid.shape, type(X_valid)))
print('valid label shape : {}\t\t dtype {}'.format(y_valid.shape, type(y_valid)))
print('test data shape   : {}\t dtype {}'.format(X_test.shape, type(X_test)))
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(5),
    transforms.ToTensor()
])

train_dataset = FashionMNIST(X_train, y_train, transform=transform)
valid_dataset = FashionMNIST(X_valid, y_valid, transform=transforms.ToTensor())
test_dataset = FashionMNIST(X_test, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_iter = iter(train_loader)

images, labels = train_iter.next()

grid = make_grid(images)

plt.rcParams['figure.figsize'] = (20, 15)
plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.axis('off');
test_iter = iter(test_loader)

images= test_iter.next()

grid = make_grid(images)

plt.rcParams['figure.figsize'] = (20, 15)
plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.axis('off');
class simpleCNN(nn.Module):
    
    def __init__(self):
        super(simpleCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.fc = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
model = simpleCNN()
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
mean_train_losses = []
mean_valid_losses = []

for epoch in range(epochs):
    model.train()
    
    train_losses = []
    valid_losses = []
    for i, (images, labels) in enumerate(train_loader):
        images, labels = Variable(images.cuda()), Variable(labels.cuda())
        
        outputs = model(images)
        
        loss = loss_fn(outputs, labels)
        train_losses.append(loss.data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i * 64 * 2) % (64 * 100) == 0:
            print(f'{i * 64 * 2} / 50000')
        
    model.eval()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(valid_loader):
        images, label = Variable(images.cuda()), Variable(labels.cuda())

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        val_losses.append(loss.data)

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
    mean_train_losses.append(np.mean(train_losses))
    mean_valid_losses.append(np.mean(valid_losses))
    
    print('epoch : {}, training loss : {}, validation loss : {}'\
          .format(epoch+1, np.mean(train_losses), np.mean(valid_losses)))
