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

from torch import nn

from sklearn.model_selection import train_test_split

import torchvision.datasets as dsets

import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
EPOCH = 10

BATCH_SIZE = 64   

TIME_STEP = 28 # image height

INPUT_SIZE = 28 #image width

LR = 0.01
train_pth = "/kaggle/input/digit-recognizer/train.csv"

test_pth = "/kaggle/input/digit-recognizer/test.csv"

train = pd.read_csv(train_pth)

test = pd.read_csv(test_pth)

print('train: {}'.format(len(train)))

print('test: {}'.format(len(test)))
y = train["label"]

X = train.drop(labels=["label"], axis=1) 



X = X.values.reshape(-1,28,28)

test = test.values.reshape(-1,28,28)
X.shape
test.shape
fig=plt.figure(figsize=(10,10))

fig.subplots_adjust(wspace=0.5, hspace=0.5)

columns = 5

rows = 5

for i in range(1, 25+1):

    fig.add_subplot(rows, columns, i)

    img = X[i]

    plt.imshow(img)

    plt.title(y[i])

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

            X = self.df.iloc[idx,1:].values.reshape((28,28)).astype(np.uint8)

            y = np.array(self.df.iloc[idx,0])

            return self.transform(X), torch.from_numpy(y)

        else:

            X = self.df.iloc[idx].values.reshape((28,28)).astype(np.uint8)

            return self.transform(X)
transform = transforms.Compose([

    transforms.ToTensor()

])
# reload

train = pd.read_csv(train_pth)

test = pd.read_csv(test_pth)



train, val = train_test_split(train, test_size = 0.2)



train_data = MNIST(train, 'train', transform)

test_data = MNIST(val, 'val', transform)



train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=20, pin_memory=True, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
class RNN(nn.Module):

    def __init__(self):

        super(RNN, self).__init__()



        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns

            input_size=INPUT_SIZE,

            hidden_size=64,         # rnn hidden unit

            num_layers=1,           # number of rnn layer

            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)

        )



        self.out = nn.Linear(64, 10)



    def forward(self, x):

        # x shape (batch, time_step, input_size)

        # r_out shape (batch, time_step, output_size)

        # h_n shape (n_layers, batch, hidden_size)

        # h_c shape (n_layers, batch, hidden_size)

        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state



        # choose r_out at the last time step

        out = self.out(r_out[:, -1, :])

        return out
rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR) 

loss_func = nn.CrossEntropyLoss() 



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

rnn.to(device)
# training and testing

for epoch in range(EPOCH):

    for step, (b_x, b_y) in enumerate(train_loader):    # gives batch data

        rnn.train()

        b_x = b_x.view(-1, 28, 28).to(device)           # reshape x to (batch, time_step, input_size)



        output = rnn(b_x)                               # rnn output

        loss = loss_func(output, b_y.to(device))        # cross entropy loss

        optimizer.zero_grad()                           # clear gradients for this training step

        loss.backward()                                 # backpropagation, compute gradients

        optimizer.step()                                # apply gradients



        if step % 100 == 0:

            rnn.eval()

            accuracy = 0.

            for step, (b_x, b_y) in enumerate(test_loader): 

                b_x = b_x.view(-1, 28, 28).to(device)  

                test_output = rnn(b_x)                   # (samples, time_step, input_size)

                pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()

                b_accuracy = float((pred_y == b_y.data.numpy()).astype(int).sum()) / float(b_y.size(0))

                accuracy += b_accuracy

                

            print('Epoch:{} | train loss: {:.4} | test accuracy: {:.4}'.format(epoch, loss.item(), accuracy/len(test_loader)))
test_x = torch.from_numpy(test.values / 255.).float();test_x.shape
test_output = rnn(test_x.view(-1, 28, 28))

pred_y = torch.max(test_output, 1)[1].data.numpy()

pred_y
submission = pd.DataFrame(np.c_[np.arange(1, 28000+1)[:,None], pred_y], 

                      columns=['ImageId', 'Label'])
submission.head()
submission.to_csv('submission.csv', index=False)