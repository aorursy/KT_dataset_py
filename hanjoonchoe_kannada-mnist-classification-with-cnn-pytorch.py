# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.optim import lr_scheduler

import torchvision

from tqdm import tqdm_notebook as tqdm
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

valid = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')

train_img = train.iloc[:,1:].astype('float').values/255.0

train_label = train.iloc[:,0].values

valid_img = valid.iloc[:,1:].astype('float').values/255.0

valid_label = valid.iloc[:,0].values
from sklearn.model_selection import train_test_split

train_img, test_img, train_label, test_label = train_test_split(train_img, train_label, test_size=0.3, random_state=7)
class Dataloader(Dataset):

    

    def __init__(self,image,label,is_train=True):

        

        self.img = image

        self.label = label

        self.is_train = is_train

        

    def __len__(self):

        return len(self.img)

    

    def __getitem__(self,idx):

        

        '''

        Reshape: 1D) 784 -> 2D) 28x28

        '''

        image1 = self.img[idx].reshape(-1,28,28)

        if self.is_train:

            label1 = np.zeros(10, dtype='float32')

            label1[self.label[idx]] =1

            return image1,label1

        else:

            return image1





trainset = Dataloader(train_img,train_label)

testset = Dataloader(test_img,test_label)

validset = Dataloader(valid_img,valid_label)



batch_size = 270



train_loader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=True)

valid_loader = torch.utils.data.DataLoader(validset,batch_size=batch_size,shuffle=True)
# I know it is too heavy :)



class Block(nn.Module):

    def __init__(self,in_channel,out_channel):

        super(Block,self).__init__()

        

        '''

        Convolution -> Max pooling -> LeakyReLU -> Dropout -> Convolution -> Max polling

        

        CHANNEL@HEIGHTxWIDTH

        

        COMPUTATION:

        H <- (H - kernel_size + 2*padding)*1/stride + 1

        W <- (W - kernel_size + 2*padding)*1/stride + 1

        CH <- out_channel

        '''

        

        self.block = nn.Sequential(

            nn.Conv2d(in_channel,out_channel,kernel_size=1),

            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),

            nn.LeakyReLU(),

            nn.Dropout(0.4),

            nn.Conv2d(out_channel,out_channel,kernel_size=3),

            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),

            #nn.BatchNorm2d(out_channel)

        )

        

    def forward(self,x):

        

        return self.block(x)

    

class CNN(nn.Module):

    def __init__(self):

        super(CNN,self).__init__()

        

        self.block1 = Block(1,32)

        self.block2 = Block(32,8)

        #self.block3 = Block(16,8)

        #self.batchnorm1 = nn.BatchNorm1d(512)

        #self.batchnorm2 = nn.BatchNorm1d(32)

        self.fc1 = nn.Linear(4608,1024)

        self.fc2 = nn.Linear(1024,512)

        self.fc3 = nn.Linear(512,128)

        self.fc4 = nn.Linear(128,32)

        self.fc5 = nn.Linear(32,10)





    def forward(self,x):

        

        x = self.block1(x)

        x = self.block2(x)

        #x = self.block3(x)

        x = x.view(x.size(0),-1)

        x = F.leaky_relu(self.fc1(x))

        x = F.leaky_relu(self.fc2(x))

        #x = self.batchnorm1(x)

        x = F.leaky_relu(self.fc3(x))

        x = F.leaky_relu(self.fc4(x))

        x = F.dropout(x)

        #x = self.batchnorm2(x)

        x = self.fc5(x)

        

        return x

net = CNN()
import torch.optim as optim

epochs = 30



'''

Adam

'''

optimizer = optim.Adam(net.parameters(), lr=0.1)



'''

# Learning_rate scheduler

'''

#scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=5,factor=0.5)



'''

Get real_time learning_rate

'''

def get_lr(optimizer):

    for param_group in optimizer.param_groups:

        return np.float(param_group['lr'])



'''

Calculate Loss (Categorical cross entropy)

'''

def criterion(input, target, size_average=True):

    """Categorical cross-entropy with logits input and one-hot target"""

    l = -(target * torch.log(F.softmax(input, dim=1) + 1e-10)).sum(1)

    if size_average:

        l = l.mean()

    else:

        l = l.sum()

    return l
'''

Source : https://github.com/Bjarten/early-stopping-pytorch

'''

class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):

        """

        Args:

            patience (int): How long to wait after last time validation loss improved.

                            Default: 7

            verbose (bool): If True, prints a message for each validation loss improvement. 

                            Default: False

            delta (float): Minimum change in the monitored quantity to qualify as an improvement.

                            Default: 0

        """

        self.patience = patience

        self.verbose = verbose

        self.counter = 0

        self.best_score = None

        self.early_stop = False

        self.val_loss_min = np.Inf

        self.delta = delta



    def __call__(self, val_loss, model):



        score = -val_loss



        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:

            self.counter += 1

            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(val_loss, model)

            self.counter = 0



    def save_checkpoint(self, val_loss, model):

        '''Saves model when validation loss decrease.'''

        if self.verbose:

            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), 'checkpoint.pt')

        self.val_loss_min = val_loss
net.train().double()

'''

Train will be automatically stopped after 7 epochs without improvement

'''

early_stopping = EarlyStopping(patience=7, verbose=True)



for epoch in range(epochs):

    i=0

    print(f'epoch: {int(epoch+1)}/{int(epochs)} || train/test/valid: {len(train_loader.dataset)}/{len(test_loader.dataset)}/{len(valid_loader.dataset)} ||' ' learning_rate: {:.4f}'.format(get_lr(optimizer)))



    for img,label in tqdm(train_loader):

        optimizer.zero_grad()

        output = net(img)

        loss = criterion(output,label.double())

        loss.backward()

        optimizer.step()

        i +=1





    net.eval()

    with torch.no_grad():

        correct = 0

        accuracy = 0

        val_correct = 0

        val_accuracy = 0

                

        for data, target in test_loader:

            output = net(data)

            pred = output.data.max(1 , keepdim=True)[1]

            correct += pred.eq(target.max(1, keepdim=True)[1].data.view_as(pred)).sum().numpy()

        accuracy = correct / len(test_loader.dataset)

                

        for data, target in valid_loader:

                    

            output = net(data)

            pred = output.data.max(1 , keepdim=True)[1]

            val_correct += pred.eq(target.max(1, keepdim=True)[1].data.view_as(pred)).sum().numpy()

        val_accuracy = correct / len(valid_loader.dataset)

        print('acc: {:.2f}%||val acc: {:.2f}%'.format(accuracy*100,val_accuracy*100))

    scheduler.step(loss)

    early_stopping(loss, net)

    if early_stopping.early_stop:

        print("Early stopping")

        break
test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

test_img = test.iloc[:,1:].astype('float').values/255.0



net.eval()

testset = Dataloader(test_img,None,is_train=False)

test_loader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False)

predictions = []



for img in tqdm(test_loader):

    

    output = net(img).max(dim=1)[1] # argmax

    predictions += list(output.data.cpu().numpy())
submission = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

submission['label'] = predictions

submission.to_csv('submission.csv', index=False)

submission.head()