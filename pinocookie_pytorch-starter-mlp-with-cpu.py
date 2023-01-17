import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

%matplotlib inline
import torch
import torch.nn as nn # 深層学習で用いるニューラルネットのコンポーネントが格納
import torch.nn.functional as F
# from torch.autograd import Variable # 自動微分用
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
print(torch.__version__)
print(os.listdir('../input/'))
class FashionMNIST(Dataset):
    """
    X : pandas.DataFrame (images)
    y : pandas.Series (labels)
    transform : list of data augmentation
    """
    def __init__(self, X, y=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __getitem__(self, index):
        # PyTorch automatically divide image data by 255 when its data type is np.uint8 
        # (np.uint8 : unchanged sign value [0 ~ 255])
        image = self.X.iloc[index, :].values.astype(np.uint8).reshape((28, 28, 1)) # (height, width, channels)
        
        if self.transform is not None:
            image = self.transform(image)
            
        if self.y is not None: # training
            return image, self.y.iloc[index]
        else: # prediction
            return image
        
    def __len__(self):
        return len(self.X.index)
train_df = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
test_df = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

print('train csv : {}'.format(train_df.shape))
print('test csv : {}'.format(test_df.shape))
test_df.drop(labels=['label'], axis=1, inplace=True)
print('test csv : {}'.format(test_df.shape))
X_train, X_valid, y_train, y_valid = \
        train_test_split(train_df.iloc[:, 1:], train_df.iloc[:, 0], test_size=1/6, random_state=0)

X_test = test_df

print('data shape on training images : {}'.format(X_train.shape))
print('data shape on training labels : {}'.format(y_train.shape))
print('data shape on validation images : {}'.format(X_valid.shape))
print('data shape on validation labels : {}'.format(y_valid.shape))
print('data shape on test images : {}'.format(X_test.shape))
transform=transforms.Compose([
    transforms.ToTensor() # PIL or numpy.ndarray を受け取りTonsor型に変換する。
])

train_dataset = FashionMNIST(X_train, y_train, transform=transform)
# 検証用・テスト用はTensor型に変換するだけで
valid_dataset = FashionMNIST(X_valid, y_valid, transform=transform)
test_dataset = FashionMNIST(X_test, transform=transform)
img, lab = train_dataset.__getitem__(0) # index=0 の画像を出力
print(img.shape)
print(lab.shape)
img.numpy().transpose((1, 2, 0)).reshape((28, 28))

plt.imshow(img.numpy().reshape((28, 28)), cmap='gray')
plt.title(f'label is {lab}');
plt.axis('off');
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
train_iter = iter(train_loader)
images, labels = next(train_iter)

print('data shape on train loader images : {}'.format(images.size()))
print('data shape on train loader labels : {}'.format(labels.size()))
print(images[0])
grid = make_grid(images, nrow=8)

fig, ax = plt.subplots(figsize=(15, 10))
ax.imshow(grid.numpy().transpose((1, 2, 0)))
ax.axis('off');
test_iter = iter(test_loader)
images = next(test_iter)

grid = make_grid(images, nrow=8)

fig, ax = plt.subplots(figsize=(15, 10))
ax.imshow(grid.numpy().transpose((1, 2, 0)))
ax.axis('off');
class MLP(nn.Module):
    
    def __init__(self):
        super(MLP, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(28*28*1, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1) # convert data shape (64, 1, 28, 28) --> (64, 1*28*28) = [64, 784]
        x = self.layers(x)
        return x
model = MLP()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.CrossEntropyLoss()
mean_train_losses = []
mean_valid_losses = []
epochs = 20

for epoch in range(epochs):
    # モデルを学習モードに変更(勾配計算やパラメータ更新を行う)
    model.train()
    
    train_losses = []
    valid_losses = []
    for i, (images, labels) in enumerate(train_loader):        
        
        # zero the parameters gradient
        optimizer.zero_grad()
        
        # feedforward -> backward -> optimize
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        train_losses.append(loss.data)
        loss.backward()
        optimizer.step()
        
        if (i * 128) % (128 * 100) == 0:
            print(f'{i * 128} / 50000')
    
    # 1epoch毎に推論を行う。
    # モデルを推論モードに変更
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(valid_loader):

            # feedforward -> loss
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            valid_losses.append(loss.data)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
    mean_train_losses.append(np.mean(train_losses))
    mean_valid_losses.append(np.mean(valid_losses))
    
    print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}, val acc : {:.2f}%'\
          .format(epoch+1, np.mean(train_losses),\
                           np.mean(valid_losses), 100*correct/total))
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(mean_train_losses, label='train')
ax.plot(mean_valid_losses, label='valid')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
model.eval()
test_preds = torch.LongTensor()

for i, images in enumerate(test_loader):

    outputs = model(images)

    pred = outputs.max(1, keepdim=True)[1]
    test_preds = torch.cat((test_preds, pred), dim=0)
out_df = pd.DataFrame()
out_df['ID'] = np.arange(1, len(X_test.index)+1)
out_df['label'] = test_preds.numpy()

out_df.head()
out_df.to_csv('submission.csv', index=None)
