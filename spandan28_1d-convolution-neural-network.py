# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torch.autograd import Variable

import torch.nn as nn

from sklearn.model_selection import train_test_split

import torch.nn.functional as F

import torch.utils.data as data_utils

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_auc_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_sets = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')



Y = data_sets["Class"]

X = data_sets.drop(columns=["Class"])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
data_sets.head(5)
class classifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.layer1 = nn.Conv1d(in_channels=1, out_channels=10,kernel_size=3,stride=3)

        self.layer2 = nn.Conv1d(in_channels=10, out_channels=5,kernel_size=2,stride=2)

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1)

        self.fc1 = nn.Linear(20,10)

        self.fc2 = nn.Linear(10,2)

        self.drop_out = nn.Dropout(0.2)

        

    def forward(self, x):

        x = x.view(-1,1,30)

        x = F.relu(self.layer1(x))

        x = self.max_pool(x)

        x = F.relu(self.layer2(x))        

        # reshape the tensor

        x = x.view(-1, 5*4)

        x = F.relu(self.fc1(x))

        x = self.drop_out(x)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

        
train_target = torch.tensor(Y_train.values.astype(np.float32))

train_data = torch.tensor(X_train.values.astype(np.float32))

train_tensor = data_utils.TensorDataset(train_data, train_target)

trainloader = data_utils.DataLoader(dataset=train_tensor, batch_size=32,shuffle=True)
test_target = torch.tensor(Y_test.values.astype(np.float32))

test_data = torch.tensor(X_test.values.astype(np.float32))

test_tensor = data_utils.TensorDataset(test_data, test_target)

testloader = data_utils.DataLoader(dataset=test_tensor, batch_size=32)
device = torch.device("cuda:0")

model = classifier().cuda()

criterion = nn.NLLLoss()

optimizer =  torch.optim.Adam(model.parameters(), lr=0.001)

print_every_n = 10

epochs= 10

model.to(device)

train_losses, test_losses = [], []

for epoch in range(epochs):

    running_loss = 0.0

    model.train()

    for step, (data, label) in enumerate(trainloader):

        data , label = data.to(device), label.to(device)

        optimizer.zero_grad()

        train_logps = model.forward(data)

        loss = criterion(train_logps, label.long())

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    else:

        with torch.no_grad():

            model.eval()

            running_test_loss = 0.0

            accuracy = 0.0

            for test_data, test_label in testloader:

                test_data, test_label = test_data.to(device), test_label.to(device)

                test_logps = model.forward(test_data)

                test_loss = criterion(test_logps, test_label.long())

                running_test_loss += test_loss.item()

                test_prob = torch.exp(test_logps)

                prob, predic_class = test_prob.topk(1, dim=1)

                equal = predic_class == test_label.long().view(*predic_class.shape)

                accuracy += torch.mean(equal.type(torch.FloatTensor)).item()

        train_losses.append(running_loss / len(trainloader))    

        test_losses.append(running_test_loss / len(testloader))                          

                

        print(f"Epoch= {epoch+1},Training loss = {running_loss/len(trainloader)},Test loss = {running_test_loss/len(testloader)}, Accuracy >> {accuracy/len(testloader)}")

                

    

    
with torch.no_grad():

    x_test = torch.tensor(X_test.values).to(device)

    predicted_value = model.forward(x_test.float()).cpu().data.numpy().argmax(axis=1)
print(f"accuracy >> {accuracy_score(predicted_value, Y_test.values)} ")

print(f"Area Under the Curve >> {roc_auc_score(predicted_value, Y_test.values)} ")
plt.plot(train_losses,color="blue",label="training error")

plt.plot(test_losses, color="red", label="test error")

plt.ylabel("Negative Log likelihood loss")

plt.xlabel("Epoch")