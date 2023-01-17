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
import pandas as pd

import numpy as np

import torch

import torch.nn as nn

import torchvision

from torchvision import transforms

from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

from sklearn.model_selection import train_test_split

from tqdm import tqdm, trange

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_pth = "/kaggle/input/digit-recognizer/train.csv"

test_pth = "/kaggle/input/digit-recognizer/test.csv"

train = pd.read_csv(train_pth)

test = pd.read_csv(test_pth)

print('train: {}'.format(len(train)))

print('test: {}'.format(len(test)))
# have a look

train.head()
test.head()
y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 

del train 

sns.countplot(y_train)
y_train.value_counts()
X_train = X_train / 255.0

test = test / 255.0



X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
X_train.shape
test.shape
fig=plt.figure(figsize=(10,10))

fig.subplots_adjust(wspace=0.5, hspace=0.5)

columns = 5

rows = 5

for i in range(1, 25+1):

    fig.add_subplot(rows, columns, i)

    img = X_train[i][:,:,0]

    plt.imshow(img)

    plt.title(y_train[i])

plt.show()
class MNIST(Dataset):

    def __init__(self, df, phase, transform=None):

        self.phase = phase

        self.df = df

        self.transform = transform

        self.phase = phase

    

    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        if self.phase in ['train', 'val']:

            X = self.df.iloc[idx,1:].values.reshape((28,28,-1)).astype(np.uint8)

            y = np.array(self.df.iloc[idx,0])

            return self.transform(X), torch.from_numpy(y)

        else:

            X = self.df.iloc[idx].values.reshape((28,28,-1)).astype(np.uint8)

            return self.transform(X)
transform = transforms.Compose([

    transforms.ToTensor(),

    # transforms.RandomHorizontalFlip(p=0.5)

])
class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(         

            nn.Conv2d(

                in_channels=1,              

                out_channels=16,            

                kernel_size=5,              

                stride=1,                   

                padding=2,                  

            ),                             

            nn.ReLU(),                     

            nn.MaxPool2d(kernel_size=2),   

        )

        self.conv2 = nn.Sequential(      

            nn.Conv2d(16, 32, 5, 1, 2),  

            nn.ReLU(),                    

            nn.MaxPool2d(2),              

        )

        self.out = nn.Linear(32 * 7 * 7, 10) 



    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = x.view(x.size(0), -1) 

        output = self.out(x)

        return output
model = CNN();model
EPOCH = 30               

BATCH_SIZE = 64

LR = 0.001 
optimizer = torch.optim.Adam(model.parameters(), lr=LR) 

criterion = nn.CrossEntropyLoss()  

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9, 15, 21], gamma=0.1)
# reload

train = pd.read_csv(train_pth)

test = pd.read_csv(test_pth)



train, val = train_test_split(train, test_size = 0.2)



train_data = MNIST(train, 'train', transform)

test_data = MNIST(val, 'val', transform)



train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=20, pin_memory=True, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
valid_loss_min = np.Inf

train_losses, valid_losses = [], []

history_accuracy = []



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)



for epoch in range(1, EPOCH+1):

    running_loss = 0

    model.train()

    with tqdm(train_loader, desc='Epoch [{}/{}]'.format(epoch, EPOCH)) as pbar1:

        for x, y in pbar1: 

            x, y = x.to(device), y.to(device)



            optimizer.zero_grad()

            output = model(x)

            loss = criterion(output, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            pbar1.set_postfix(loss=loss.item(), refresh=True)

        train_losses.append(running_loss / len(train_loader))

        lr_scheduler.step()

        

    with torch.no_grad():

        with tqdm(test_loader, desc='Testing') as pbar2:

            model.eval()

            accuracy = 0

            valid_loss = 0

            for x, y in pbar2:

                x, y = x.to(device), y.to(device)

                test_output = model(x)

                batch_loss = criterion(test_output, y)

                valid_loss += batch_loss

                pred_y = torch.max(test_output, 1)[1].data.squeeze()

                batch_correct = pred_y.eq(y.data.view_as(pred_y)).cpu().sum()



                batch_acc = batch_correct.item() / y.size(0)

                pbar2.set_postfix(loss=batch_loss.item(),acc=batch_acc, refresh=True)

                accuracy += batch_acc

            valid_losses.append(valid_loss / len(test_loader))    

            history_accuracy.append(accuracy / len(test_loader))

            

            # save the best model

            if valid_losses[-1] < valid_loss_min:

                torch.save(model.state_dict(), "./model.pth")

import matplotlib.pyplot as plt



plt.plot(train_losses, label='Training Loss')

plt.plot(valid_losses, label='Validation Loss')

plt.legend(frameon=False)
plt.plot(history_accuracy, label='Validation Accuracy')

plt.legend(frameon=False)
# load model

model = CNN()

model.load_state_dict(torch.load('model.pth'))

model.eval()
dataset = MNIST(test, 'test', transform)



dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=28000, num_workers=20)



model.to(device)

for i, data in enumerate(tqdm(dataloader)):

    data = data.to(device)

    output = model(data)

    pred_y = torch.max(output, 1)[1].cpu().data.numpy().squeeze()
submission = pd.DataFrame(np.c_[np.arange(1, len(dataset)+1)[:,None], pred_y], 

                      columns=['ImageId', 'Label'])
submission.head()
submission.to_csv('submission.csv', index=False)