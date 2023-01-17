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
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import pandas as pd
%matplotlib inline
train = pd.read_csv("../input/digit-recognizer/train.csv",dtype = np.float32)
test = pd.read_csv("../input/digit-recognizer/test.csv",dtype = np.float32)
print(train.shape)
print(test.shape)
train.head()
test.head()
targets = train.label.values # 1st column that is values
features = train.drop(labels = ["label"],axis = 1).values / 255 #normalize
targetsTrain = torch.from_numpy(targets).type(torch.LongTensor)
featuresTrain = torch.from_numpy(features)
trn = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
val_size = 4200
train_size = len(trn) - val_size

train_ds, val_ds = random_split(trn, [train_size, val_size])
len(train_ds), len(val_ds)
batch_size = 256
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
for images,labels in train_loader:
    plt.imshow(images[28].reshape(28,28))
    plt.axis("off")
    plt.title(str(labels[28]))
    plt.show()
    break
for images, _ in train_loader:
    imgs = []
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    for img in images:
        im = img.reshape(28,28)
        im = im.view(-1,im.size(0),im.size(1))
        imgs.append(im)
        plt.axis('off')
    plt.imshow(make_grid(imgs, nrow=16).permute((1,2,0)))
    break
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
class Mnistnn(nn.Module):
    def __init__(self,in_size,hidden_size1,hidden_size2,hidden_size3,out_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2,hidden_size3)
        self.linear4 = nn.Linear(hidden_size3,out_size)
    
    def forward(self,xb):
        #xb.size(0) will be the batch size(change dynamically)
        # -1 ths will be calculate by pytorch itself..
        xb = xb.view(-1,xb.size(1))
        out = self.linear1(xb)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        return out
    
    def training_step(self,batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out,labels)
        return loss
    
    def validation_step(self,batch):
        images,labels = batch
        out = self(images)
        loss = F.cross_entropy(out,labels)
        acc = accuracy(out,labels)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self,outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self,epoch,result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
input_size = 784
hidden_size1 = 512
hidden_size2 = 256
hidden_size3 = 64
num_classes = 10
model = Mnistnn(input_size, hidden_size1=hidden_size1, hidden_size2= hidden_size2,hidden_size3 = hidden_size3, out_size=num_classes)
for t in model.parameters():
    print(t.shape)
for images, labels in train_loader:
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    print('Loss:', loss.item())
    break

print('outputs.shape : ', outputs.shape)
print('Sample outputs :\n', outputs[:2].data)
torch.cuda.is_available()
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()
device
# Moving PyTorch Tensors to GPU
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
for images, labels in train_loader:
    print(images.shape)
    images = to_device(images, device)
    print(images.device)
    break
class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)
train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)
for xb, yb in val_loader:
    print('xb.device:', xb.device)
    print('yb:', yb)
    break
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    time = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        time.append(result)
    return time
model = Mnistnn(input_size, hidden_size1=hidden_size1, hidden_size2= hidden_size2, hidden_size3= hidden_size3, out_size=num_classes)
to_device(model, device)
time = [evaluate(model, val_loader)]
time
time += fit(10, 0.5, model, train_loader, val_loader) #Learning Rate = 0.5
time += fit(16, 0.2, model, train_loader, val_loader)
time += fit(16, 0.0001, model, train_loader, val_loader)
losses = [x['val_loss'] for x in time]
plt.plot(losses, '-b')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss vs. No. of epochs');
accuracies = [x['val_acc'] for x in time]
plt.plot(accuracies, 'r-')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');
features_test = test.values / 255
features_tensor = torch.from_numpy(features_test)
test_loader = DataLoader(features_tensor, batch_size*2, num_workers=4, pin_memory=True)
# Prediction Function
def predict(tl, model):
    preds = torch.cuda.LongTensor()
    for img in tl:
        y = model(img)
        _, pred  = torch.max(y, dim=1)
        preds = torch.cat((preds, pred), dim=0)
    return preds
for images in test_loader:
    imgs = []
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,9))
    for img in images:
        im = img.reshape(28,28)
        im = im.view(-1,im.size(0),im.size(1))
        imgs.append(im)
        plt.axis('off')
    plt.imshow(make_grid(imgs, nrow=32).permute((1,2,0)))
    break
test_loader_gpu = DeviceDataLoader(test_loader, device)
test_preds = predict(test_loader_gpu,model)
test_predictions = test_preds.cpu()
test_df = pd.DataFrame(np.c_[np.arange(1, len(test)+1), test_predictions.numpy()], 
                      columns=['ImageId', 'Label'])
test_df.to_csv('submission.csv', index=False)
test_df.head()
