import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
print("Reading the data...")
data_raw = pd.read_csv('../input/train.csv', sep=",")
test_data_raw = pd.read_csv('../input/test.csv', sep=",")

print("Reshaping the data...")
labels_raw = data_raw['label']
data_raw = data_raw.drop('label', axis=1)

data = data_raw.values
labels = labels_raw.values
test_data = test_data_raw.values

print("Data is ready")
def reshape_to_2d(data):
    a = []
    for i in data:
        a.append(i.reshape(1, 28, 28))

    return np.array(a)
data = reshape_to_2d(data)
test_data = reshape_to_2d(test_data)
x = torch.FloatTensor(data)
y = torch.LongTensor(labels.tolist())
epochs = 20
batch_size = 50
learning_rate = 0.00003
class Network(nn.Module): 
    
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.dropout1 = nn.Dropout2d()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(10, 25, 3)
        self.dropout2 = nn.Dropout2d()
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(25, 50, 3)
        
        #self.fc3 = nn.Linear(25 * 4 * 4, 250)
        self.fc3 = nn.Linear(50 * 3 * 3, 250)
        self.fc4 = nn.Linear(250, 120)
        self.fc5 = nn.Linear(120, 10)
        
        self.softmax = nn.LogSoftmax(dim=1)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        
        #print(x.shape)
        
        x = x.view(-1, 50 * 3 * 3) 
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        
        #return F.log_softmax(x, dim=1)
        return self.softmax(x)
    
net = Network()
print(net)
#optimizer = optim.SGD(net.parameters(), learning_rate, momentum=0.9)
optimizer = optim.Adam(net.parameters(), learning_rate)
loss_func = nn.CrossEntropyLoss()
loss_log = []

for e in range(epochs):
    for i in range(0, x.shape[0], batch_size):
    #for i in range(0, x.item(), batch_size):
        x_mini = x[i:i + batch_size] 
        y_mini = y[i:i + batch_size] 
        
        optimizer.zero_grad()
        net_out = net(Variable(x_mini))
        
        loss = loss_func(net_out, Variable(y_mini))
        loss.backward()
        optimizer.step()
        
        if i % 1000 == 0:
            loss_log.append(loss.item())
        
    print('Epoch: {} - Loss: {:.6f}'.format(e + 1, loss.item()))
plt.figure(figsize=(10,8))
plt.plot(loss_log)
test = torch.FloatTensor(test_data)
test_var = Variable(test)

net_out = net(test_var)
output = (torch.max(net_out.data, 1)[1]).numpy()

np.savetxt("submission.csv", np.dstack((np.arange(1, output.size+1),output))[0],\
           "%d,%d",header="ImageId,Label", comments='')