import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
%matplotlib inline
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
print(torch.__version__)
print(torchvision.__version__)
class FashionMNIST(Dataset):
    
    def __init__(self, X, y=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X.index)
    
    def __getitem__(self, index):
        image = self.X.iloc[index, :].values.astype(np.uint8).reshape((28, 28, 1))
        
        if self.transform is not None:
            image = self.transform(image)
            
        if self.y is not None:
            return image, self.y.iloc[index]
        else:
            return image
train_df = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
test_df = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

print('train data shape ; ', train_df.shape)
print('test data shape : ', test_df.shape)
X_train, X_valid, y_train, y_valid = \
    train_test_split(train_df.iloc[:, 1:], train_df['label'], test_size=1/6, random_state=42)

X_test = test_df.iloc[:, 1:]

print('train image shape :', X_train.shape)
print('train label shape :', y_train.shape)
print('valid image shape :', X_valid.shape)
print('valid label shape :', y_valid.shape)
print('test image shape :', X_test.shape)
transform = transforms.Compose([
    transforms.ToTensor()
])

train_ds = FashionMNIST(X=X_train, y=y_train, transform=transform)
valid_ds = FashionMNIST(X=X_valid, y=y_valid, transform=transform)
test_ds = FashionMNIST(X=X_test, transform=transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=128, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=100, shuffle=False)
dataiter = iter(train_loader)
images, labels = dataiter.next()

print('images tensor shape :', images.size())
print('labels tensor shape :', labels.size())

grid = make_grid(images, nrow=12)

fig, ax = plt.subplots(figsize=(18, 15))
ax.imshow(grid.numpy().transpose((1, 2, 0)))
ax.axis('off')
def mixup_data(inputs, labels):
    """
    inputs : minibatch tensor inputs shape --> (minibatch, channels, height, width)
    labels : target shape --> (minibatch)
    
    return mixed_inputs --> (minibatch, channels, height, width)
           mixed_labels --> (minibatch, num_classes)
    """
    batch_size = inputs.size(0)
    rnd_idx1 = torch.randperm(batch_size) # --> random series [0, N-1] 
    inputs1 = inputs[rnd_idx1]
    labels1 = labels[rnd_idx1]        # --> (minibatch)
    labels1_2d = labels1.unsqueeze(1) # --> (minibatch, 1)
    
    rnd_idx2 = torch.randperm(batch_size) # --> random series [0, N-1] 
    inputs2 = inputs[rnd_idx2]
    labels2 = labels[rnd_idx2]        # --> (minibatch)
    labels2_2d = labels2.unsqueeze(1) # --> (minibatch, 1)
    
    # initialize one-hot matrix --> (minibatch, num_classes)
    # -> create one-hot torch.scatter_
    y_onehot_1 = torch.FloatTensor(batch_size, 10).zero_()
    labels1_onehot = y_onehot_1.scatter_(1, labels1_2d, 1)
    
    y_onehot_2 = torch.FloatTensor(batch_size, 10).zero_()
    labels2_onehot = y_onehot_2.scatter_(1, labels2_2d, 1)
    
    # take paramter alpha from beta distribution
    a = np.random.beta(1, 1, [batch_size, 1])
    
    # expand alpha parameter to image shape
    # a[..., None, None] --> [128, 1, 1, 1] --> [1, 1, 28, 28] with each image on minibatch --> [128, 1, 28, 28]
    b = np.tile(a[..., None, None], [1, 1, 28, 28])
    
    # mixup images
    inputs1 = inputs1.float() * torch.from_numpy(b).float() # element-wise multiply
    inputs2 = inputs2.float() * torch.from_numpy(1-b).float()
    
    # expand alpha parameter to label shape
    # a --> [minibatch, 1] --> [1, 10] with each label on minibatch --> [128, 10]
    c = np.tile(a, [1, 10])
    
    labels1 = labels1_onehot.float() * torch.from_numpy(c).float()
    labels2 = labels2_onehot.float() * torch.from_numpy(1-c).float()
    
    inputs_mixup = inputs1 + inputs2
    labels_mixup = labels1 + labels2
    
    return inputs_mixup, labels_mixup
dataiter = iter(train_loader)
images, labels = dataiter.next()
images, labels = mixup_data(images, labels)

print('images tensor shape :', images.size())
print('labels tensor shape :', labels.size())

grid = make_grid(images, nrow=12)

fig, ax = plt.subplots(figsize=(18, 15))
ax.imshow(grid.numpy().transpose((1, 2, 0)))
ax.axis('off')
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
        images, labels = mixup_data(images, labels)
        # zero the parameters gradient
        optimizer.zero_grad()
        
        # feedforward -> backward -> optimize
        outputs = model(images)
        m = nn.LogSoftmax()
        loss = -m(outputs) * labels
        loss = torch.sum(loss) / 128
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.data)
        
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
