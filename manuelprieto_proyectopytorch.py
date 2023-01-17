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
import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm.notebook import tqdm

import torch

import torch.nn.functional as F

from torch.optim import lr_scheduler

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split



from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable



import torchvision

import torchvision.transforms as transforms



import math

import random



%matplotlib inline
train_df = pd.read_csv('../input/digit-recognizer/train.csv')

n_pixels = len(train_df.columns) - 1
BATCH_SIZE = 64

EPOCHS = 50

VALID_SIZE = 0.15 # percentage of training set to use as validation
# Dataset responsible for manipulating data for training as well as training tests.z

class DatasetMNIST(torch.utils.data.Dataset):

    def __init__(self, file_path, 

                 transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), 

                     transforms.Normalize(mean=(0.5,), std=(0.5,))])

                ):

        df = pd.read_csv(file_path)

        

        if len(df.columns) == n_pixels:

            # test data

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
transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.RandomRotation(0, 0.5),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])



train_data = DatasetMNIST('../input/digit-recognizer/train.csv', transform= transform)



valid_train_data = DatasetMNIST('../input/digit-recognizer/train.csv', transform= transform)



test_data = DatasetMNIST('../input/digit-recognizer/test.csv', transform= transform)





# Shuffling data and choosing data that will be used for training and validation

num_train = len(train_data)

indices = list(range(num_train))

np.random.shuffle(indices)

split = int(np.floor(VALID_SIZE * num_train))

train_idx, valid_idx = indices[split:], indices[:split]



train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)





train_loader = torch.utils.data.DataLoader(dataset=train_data,

                                           batch_size=BATCH_SIZE,sampler=train_sampler)



valid_loader = torch.utils.data.DataLoader(dataset=valid_train_data,

                                           batch_size=BATCH_SIZE,sampler=valid_sampler)



test_loader = torch.utils.data.DataLoader(dataset=test_data,

                                           batch_size=BATCH_SIZE, shuffle=False)
class CNN(nn.Module):

        def __init__(self):

            super(CNN,self).__init__()



            self.conv = nn.Sequential(

                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),

                nn.BatchNorm2d(32),

                nn.ReLU(inplace=True),

                

                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),

                nn.BatchNorm2d(32),

                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Dropout(p = 0.25),

                

                nn.Conv2d(32, 64, kernel_size=3, padding=1),

                nn.BatchNorm2d(64),

                nn.ReLU(inplace=True),

                

                nn.Conv2d(64, 64, kernel_size=3, padding=1),

                nn.BatchNorm2d(64),

                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Dropout(p = 0.25),

            )



            self.fc = nn.Sequential(

                nn.Linear(64 * 7 * 7, 512),

                nn.BatchNorm1d(512),

                nn.ReLU(inplace=True),

                nn.Dropout(p = 0.5),

                nn.Linear(512, 10),

            )

            

            #Weight Initilitation 

            for m in self.conv.children():

                if isinstance(m, nn.Conv2d):

                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

                    m.weight.data.normal_(0, math.sqrt(2. / n))

                elif isinstance(m, nn.BatchNorm2d):

                    m.weight.data.fill_(1)

                    m.bias.data.zero_()



            for m in self.fc.children():

                if isinstance(m, nn.Linear):

                    nn.init.xavier_uniform_(m.weight)

                elif isinstance(m, nn.BatchNorm1d):

                    m.weight.data.fill_(1)

                    m.bias.data.zero_()





        def forward(self, x):

            x= self.conv(x)

            x = x.view(x.size(0), -1)

            x = self.fc(x)



            return F.log_softmax(x, dim=1)  







model = CNN()

print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

error = nn.CrossEntropyLoss()
if torch.cuda.is_available():

    model = model.cuda()

    error = error.cuda()


def fit(epoch):

    

    valid_loss_min = np.Inf

    running_loss= 0

    accuracy_train = 0

    model.train() 

    exp_lr_scheduler.step()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = Variable(data), Variable(target)

            

        if torch.cuda.is_available():

            data = data.cuda()

            target = target.cuda()



        optimizer.zero_grad()

        output = model(data)

        loss = error(output, target)

  

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        

        _, top_class = output.topk(1, dim=1)

                

        equals = top_class == target.view(*top_class.shape)

        

        accuracy_train += torch.mean(equals.type(torch.FloatTensor))

        

    valid_loss = 0

    accuracy_val = 0

        

    with torch.no_grad():

               

        model.eval() 

        for data, target in valid_loader:

            data, target = Variable(data), Variable(target)

                

            if  torch.cuda.is_available():

                data, target = data.cuda(), target.cuda()

               

            output = model(data)

                



            _, top_class = output.topk(1, dim=1)

                

            equals = top_class == target.view(*top_class.shape)

                

            loss = error(output, target)

            valid_loss += loss.item() 

            accuracy_val += torch.mean(equals.type(torch.FloatTensor))

                

    model.train() 

        

    train_losses.append(running_loss/len(train_loader))

    valid_losses.append(valid_loss/len(valid_loader))

        

    history_accuracy_val.append(accuracy_val/len(valid_loader))

    history_accuracy_train.append(accuracy_train/len(train_loader))

        

    network_learned = valid_loss < valid_loss_min



   

    if epoch == 1 or epoch % 5 == 0 or network_learned:

        print(f"Epoch: {epoch}/{EPOCHS}.. ",

               f"Training Loss: {running_loss/len(train_loader):.3f}.. ",

                f"Validation Loss: {valid_loss/len(valid_loader):.3f}.. ",

                 f"Training Accuracy: {accuracy_train/len(train_loader):.3f}.. ",

                f"Validation Accuracy: {accuracy_val/len(valid_loader):.3f}")

        

    if network_learned:

        valid_loss_min = valid_loss

        torch.save(model.state_dict(), 'model_mtl_mnist.pt')

        print('Detected network improvement, saving current model')

        
train_losses, valid_losses = [], []

history_accuracy_train,history_accuracy_val = [], []

for epoch in range(EPOCHS):

    fit(epoch)
plt.plot(train_losses, label='Training Loss')

plt.plot(valid_losses, label='Validation Loss')

plt.legend(frameon=False)
plt.plot(history_accuracy_train, label='Accuracy Train')

plt.plot(history_accuracy_val, label='Accuracy Validation')

plt.legend(frameon=False)
def prediciton():

    model.eval()

    test_pred = torch.LongTensor()

    for i,data in enumerate(test_loader):

        data = Variable(data, requires_grad=True)

        if torch.cuda.is_available():

            data = data.cuda()

            

        output = model(data)

        

        pred = output.cpu().data.max(1, keepdim=True)[1]

        test_pred = torch.cat((test_pred, pred), dim=0)

        

    return test_pred
test_pred = prediciton()  

out_df = pd.DataFrame(np.c_[np.arange(1, len(test_data)+1)[:,None], test_pred.numpy()], 

                      columns=['ImageId', 'Label'])



print(out_df)
out_df.to_csv('submission.csv', index=False)