# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pickle

print(torch.__version__)
# 定义transform
transform = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,),)
])
# 读取数据
train_set = datasets.FashionMNIST('./', download=True, train=True, transform=transform)
test_set = datasets.FashionMNIST('./', download=True, train=False, transform=transform)
# 验证集大小
val_size = int(len(train_set) * 0.2)
val_size
train_set, val_set = torch.utils.data.random_split(train_set, [len(train_set)-val_size, val_size])
len(train_set), len(val_set)
# 设置batch_size大小
BATCH_SIZE = 64
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
val_loader = torch.utils.data.DataLoader(val_set, shuffle=True, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=BATCH_SIZE)
data_iter = iter(train_loader)
images, labels = data_iter.next()
images.size(), labels.size(), labels[:5]
img, _ = next(iter(train_set))
plt.imshow(img.squeeze(), cmap='gray')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

# 定义网络
class LRNet(nn.Module):
    def __init__(self):
        super(LRNet, self).__init__();
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.leaky_relu(self.fc1(x)))
        x = self.dropout(F.leaky_relu(self.fc2(x)))
        x = self.dropout(F.leaky_relu(self.fc3(x)))
        x = self.fc4(x)
        return x
class BestAcc:
    def __init__(self, epoch, loss, acc):
        self.epoch = epoch
        self.loss = loss
        self.acc = acc
    
    def update(self, epoch, loss, acc):
        if acc > self.acc:
            self.epoch = epoch
            self.loss = loss
            self.acc = acc
            
class MyModel:

    def __init__(self, train_loader, val_loader, test_loader, model, criterion, optimizer, batch_size, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.device = device

        self.train_loss_arr = []
        self.val_loss_arr = []
        self.train_acc_arr = []
        self.val_acc_arr = []
        self.train_log = ""
        self.val_log = ""
        self.test_log = ""
        self.train_best = BestAcc(-1, 0, -float('inf'))
        self.val_best = BestAcc(-1, 0, -float('inf'))
        self.epochs = 0
        self.model = model
        self.is_model_loaded = False

    def train_and_val(self, epochs, force_train=False):
        self.epochs = epochs

        if self.is_model_loaded and not force_train:
            return False

        for epoch in range(epochs):
            train_loss = 0.0
            val_loss = 0.0
            train_correct = 0
            val_correct = 0
            for i, (images, labels) in enumerate(self.train_loader, 0):
                images, labels = images.to(self.device), labels.to(self.device)

                pred = self.model(images)
                loss = self.criterion(pred, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss
                label_pred = pred.max(1)[1]
                train_correct += label_pred.eq(labels.data).sum()
                if i % 100 == 0:
                    log = 'Train epoch #{} {}/{} {:.0f}%\tLoss: {:.6f}\n' \
                        .format(epoch, i * self.batch_size, len(self.train_loader.dataset),
                                100.0 * i / len(self.train_loader), loss.item())
                    print(log)
                    self.train_log += log

            train_loss_ave = train_loss / len(self.train_loader)
            self.train_loss_arr.append(train_loss_ave)
            train_acc = 1.0 * train_correct / len(self.train_loader.dataset)
            self.train_acc_arr.append(train_acc)
            self.train_best.update(epoch, train_loss_ave, train_acc)

            #
            for i, (images, labels) in enumerate(self.val_loader, 0):
                images, labels = images.to(self.device), labels.to(self.device)
                pred = self.model(images)

                loss = self.criterion(pred, labels)
                val_loss += loss

                label_pred = pred.max(1)[1]
                val_correct += label_pred.eq(labels.data).sum()

            val_loss_ave = val_loss / len(self.val_loader)
            self.val_loss_arr.append(val_loss_ave)
            val_acc = 1.0 * val_correct / len(self.val_loader.dataset)
            self.val_acc_arr.append(val_acc)
            self.val_best.update(epoch, val_loss_ave, val_acc)
            log = 'Validation epoch #{} Average loss: {:.4f}\tAccuracy: {}/{} {:.2f}%\n' \
                .format(epoch, val_loss_ave, val_correct, len(self.val_loader.dataset), 100.0 * val_acc)
            print(log)
            self.val_log += log
        return True

    def test(self):
        test_loss = 0.0
        test_correct = 0
        for i, (images, labels) in enumerate(self.test_loader, 0):
            images, labels = images.to(self.device), labels.to(self.device)
            pred = self.model(images)

            loss = self.criterion(pred, labels)
            test_loss += loss

            label_pred = pred.max(1)[1]
            test_correct += label_pred.eq(labels.data).sum()

        test_loss_ave = test_loss / len(self.test_loader)

        test_acc = 1.0 * test_correct / len(self.test_loader.dataset)

        log = 'Test Average loss: {:.4f}\tAccuracy: {}/{} {:.2f}%\n' \
            .format(test_loss_ave, test_correct, len(self.test_loader.dataset), 100.0 * test_acc)
        print(log)
        self.test_log += log

    def get_loss_data(self):
        return self.train_loss_arr, self.val_loss_arr

    def get_acc_data(self):
        return self.train_acc_arr, self.val_acc_arr

    def get_log(self):
        return self.train_log, self.val_log, self.test_log

    def get_best_acc(self):
        return self.train_best, self.val_best

    def draw_acc(self):
        x_axis = np.arange(self.epochs)
        plt.figure()
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.plot(x_axis, self.train_acc_arr, label='train')
        plt.plot(x_axis, self.val_acc_arr, label='val')
        plt.legend()
        plt.show()

    def draw_loss(self):
        x_axis = np.arange(self.epochs)
        plt.figure()
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.plot(x_axis, self.train_loss_arr, label='train')
        plt.plot(x_axis, self.val_loss_arr, label='val')
        plt.legend()
        plt.show()

    def load_model(self, path_model):
        self.model.load_state_dict(torch.load(path_model))
        self.model.eval()
        self.is_model_loaded = True
        return True

    def save_model(self, path_model):
        torch.save(self.model.state_dict(), path_model)


LEARNING_RATE = 0.0001

net = LRNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

EPOCHS = 45
%%time
LRModel = MyModel(train_loader, val_loader, test_loader, net, criterion, optimizer, BATCH_SIZE, device)
# LRModel.load_model('/kaggle/working/linear_model.pkl')
LRModel.train_and_val(EPOCHS)
LRModel.save_model('/kaggle/working/linear_model.pkl')
LRModel.test()
train_best, val_best = LRModel.get_best_acc()
print('Train best epoch: {}, loss: {:.4f}, accuracy: {:.4f}'.format(train_best.epoch, train_best.loss, train_best.acc))
print('Val best epoch: {}, loss: {:.4f}, accuracy: {:.4f}'.format(val_best.epoch, val_best.loss, val_best.acc))
LRModel.draw_loss()
LRModel.draw_acc()
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(64*6*6, 600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, 10)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x
# 设置batch_size大小
BATCH_SIZE = 32
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
val_loader = torch.utils.data.DataLoader(val_set, shuffle=True, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=BATCH_SIZE)
LEARNING_RATE = 0.001
cnn = CNN().to(device)
EPOCHS = 5

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
print(cnn)
%%time
CNNModel = MyModel(train_loader, val_loader, test_loader, cnn, criterion, optimizer, BATCH_SIZE, device)
# CNNModel.load_model('/kaggle/working/cnn_model.pkl', '/kaggle/working/cnn_model_data.pkl')
CNNModel.train_and_val(EPOCHS)
CNNModel.save_model('/kaggle/working/cnn_model.pkl', '/kaggle/working/cnn_model_data.pkl')
CNNModel.test()
train_best, val_best = CNNModel.get_best_acc()
print('Train best epoch: {}, loss: {:.4f}, accuracy: {:.4f}'.format(train_best.epoch, train_best.loss, train_best.acc))
print('Val best epoch: {}, loss: {:.4f}, accuracy: {:.4f}'.format(val_best.epoch, val_best.loss, val_best.acc))
CNNModel.draw_loss()
CNNModel.draw_acc()