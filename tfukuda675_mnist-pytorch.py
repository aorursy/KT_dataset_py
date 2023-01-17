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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
%matplotlib inline

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
### => train/testの違い
train_df  = pd.read_csv("../input/digit-recognizer/train.csv") 
test_df = pd.read_csv("../input/digit-recognizer/test.csv") 
train_df.shape
train_df.head()
test_df.shape
test_df.head()
random_n = np.random.randint(len(train_df),size=8)
random_n

### => なぜdataframeにmaxを2回も
train_df.max().max()  ##255

### => 255で割っている理由
### => reshapeと-1について
torch.Tensor((train_df.iloc[random_n, 1:].values/255.).reshape((-1,28,28)))

## 4次元へ変換
### => unsqueezeについて
torch.Tensor((train_df.iloc[random_n, 1:].values/255.).reshape((-1,28,28))).unsqueeze(1)

### => make_grid
grid = make_grid(torch.Tensor((train_df.iloc[random_n, 1:].values/255.).reshape((-1, 28, 28))).unsqueeze(1), nrow=8)
plt.rcParams['figure.figsize'] = (16, 2)

### => transposeについて
plt.imshow(grid.numpy().transpose((1,2,0)))
plt.axis('off')
print(*list(train_df.iloc[random_n, 0].values), sep = ', ')
train_df.iloc[:,1:].mean(axis=1).mean()
train_df.iloc[:,1:]

### => Datasetについて
class MNISTDataset(Dataset):
    """MNIST dtaa set"""
    
    def __init__(self, dataframe, 
                 ### => transforms.Composeについて
                 transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.ToTensor(),
                                                 ### => meanとstdの確認
                                                 transforms.Normalize(mean=(0.5,), std=(0.5,))])
                ):
        df = dataframe
        # for MNIST dataset n_pixels should be 784
        self.n_pixels = 784
        
        if len(df.columns) == self.n_pixels:
            # test data　次の方法で次元を増やしている。
            self.X = df.values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
            self.y = None
        else:
            # training data
            self.X = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
            self.y = torch.from_numpy(df.iloc[:,0].values)
            
        self.transform = transform
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.transform(self.X[idx])
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

### => MNIST用にResNetを修正
class MNISTResNet(ResNet):
    def __init__(self):
        #super(MNISTResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10) # Based on ResNet18
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=10) # Based on ResNet18
        # super(MNISTResNet, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=10) # Based on ResNet34
        # super(MNISTResNet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=10) # Based on ResNet50
        # Conv2dのoutput_channel 64 はデフォルト

        ### => conv1のconv2dについて
        ### => kernelについて
        ### => strideについて
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,bias=False)
        #self.fc    = nn.Linear(784 * 1, 10)

model = MNISTResNet()
print(model)
### => trainについて
def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    loss_train = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # if GPU available, move data and target to GPU
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        # compute output and loss
        output = model(data)
        loss = criterion(output, target)
        #loss_train += criterion(output, target).data.item()
        
        # TODO:
        # 1. add batch metric (acc1, acc5)
        # 2. add average metric top1=sum(acc1)/batch_idx, top5 = sum(acc5)/batch_idx
        
        # backward and update model
        ### => zero_grad()
        optimizer.zero_grad()
        ### => loss.backward()
        loss.backward()
        ### => optimizer.step()
        optimizer.step()
        
        if (batch_idx + 1)% 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.data.item()))
    
    #loss_train /= len(val_loader.dataset)
    #history['train_loss'].append(loss_train)
def validate(val_loader, model, criterion):
    model.eval()
    loss = 0
    correct = 0
    
    for _, (data, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        ### => modelにdataを入力したとき
        output = model(data)
        
        ### => validate時のlossについて
        loss += criterion(output, target).data.item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        
        
    loss /= len(val_loader.dataset)
    #history['val_loss'].append(loss)
        
    print('\nOn Val set Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(val_loader.dataset),
        100.0 * float(correct) / len(val_loader.dataset)))

train_transforms = transforms.Compose(
    [transforms.ToPILImage(),
### => RandAffineについて
#     RandAffine,
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.5,), std=(0.5,))])

val_test_transforms = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.5,), std=(0.5,))])
# example config, use the comments to get higher accuracy
total_epoches = 20 # 50
step_size = 5     # 10
base_lr = 0.01    # 0.01
batch_size = 64


### => optimizerについて
optimizer = optim.Adam(model.parameters(), lr=base_lr)
### => nn.CrossEntropyLossについて
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
def split_dataframe(dataframe=None, fraction=0.9, rand_seed=1):
    df_1 = dataframe.sample(frac=fraction, random_state=rand_seed)
    df_2 = dataframe.drop(df_1.index)
    return df_1, df_2
#history = {
#    'train_loss': [],
#    'val_loss': [],
#    'test_acc': [],
#}


for epoch in range(total_epoches):
    print("\nTrain Epoch {}: lr = {}".format(epoch, exp_lr_scheduler.get_lr()[0]))

    train_df_new, val_df = split_dataframe(dataframe=train_df, fraction=0.9, rand_seed=epoch)
    
    ## オリジナルではget_datasetでMNISTDatasetをwrappingしているが、わかりにくいので外す。
    #train_dataset = get_dataset(train_df_new, transform=train_transforms)
    #val_dataset = get_dataset(val_df, transform=val_test_transforms)
    train_dataset = MNISTDataset(train_df_new, transform=train_transforms)
    val_dataset = MNISTDataset(val_df, transform=val_test_transforms)
    
    
    # CIFAR10の場合、次。torchvisionにデータが準備されているのでそれを利用するだけ。
    #train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    #test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size, shuffle=False)
    
    

    train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch)
    validate(val_loader=val_loader, model=model, criterion=criterion)
    exp_lr_scheduler.step()

#plt.figure()
#plt.plot( range(1, total_epoches+1), history['train_loss'], label='train_loss' )
#plt.plot( range(1, total_epoches+1), history['val_loss'], label='val_loss' )
#plt.show()
def prediciton(test_loader, model):
    model.eval()
    test_pred = torch.LongTensor()
    
    for i, data in enumerate(test_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            
        output = model(data)
        
        pred = output.cpu().data.max(1, keepdim=True)[1]
        test_pred = torch.cat((test_pred, pred), dim=0)
        
    return test_pred
x = torch.tensor(2.5)
x
x.data
x.cpu().data
test_batch_size = 64
## オリジナルではget_datasetでMNISTDatasetをwrappingしているが、わかりにくいので外す。
#test_dataset = get_dataset(test_df, transform=val_test_transforms)
test_dataset = MNISTDataset(test_df, transform=val_test_transforms)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=test_batch_size, shuffle=False)

# tensor prediction
test_pred = prediciton(test_loader, model)

# tensor -> numpy.ndarray -> pandas.DataFrame
test_pred_df = pd.DataFrame(np.c_[np.arange(1, len(test_dataset)+1), test_pred.numpy()], 
                      columns=['ImageId', 'Label'])

# show part of prediction dataframe
print(test_pred_df.head())
# import the modules we'll need
from IPython.display import HTML
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a link and click to download
create_download_link(test_pred_df, filename="submission.csv")
test_pred_df.to_csv('submission.csv', index=False)