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
data_file = pd.read_csv('../input/digit-recognizer/train.csv')
data_file.head()
import matplotlib.pyplot as plt
Y = data_file.iloc[:,0].values
X = data_file.iloc[:,1:].values
X = X/255
for i in range(8):
    plt.subplot(4,2,i+1)
    plt.title(Y[i])
    plt.imshow(X[i,:].reshape(28,28))
plt.show()
Y.shape
X.shape
import torch 
import torch.nn as nn
torch.cuda.is_available()
device = 'cuda:0'
X_test = X[0:36000,:]
Y_test = Y[0:36000]
X_vald = X[36000:,:]
Y_vald = Y[36000:]
class NeuralNet(nn.Module):
    def __init__(self,input_features,hidden_size,classes):
        super().__init__()
        self.linear1 = nn.Linear(input_features,hidden_size[0])
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size[0],hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1],hidden_size[2])
        self.linear4 = nn.Linear(hidden_size[2],hidden_size[3])
        self.linear5 = nn.Linear(hidden_size[3],hidden_size[4])
        self.linear6 = nn.Linear(hidden_size[4],hidden_size[5])
        self.linear7 = nn.Linear(hidden_size[5],classes)
    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        out = self.relu(out)
        out = self.linear5(out)
        out = self.relu(out)
        out = self.linear6(out)
        out = self.relu(out)
        out = self.linear7(out)
        return out
        
input_size = X.shape[1]
categories = 10
hidden_layers = (700,500,400,250,64,64)
model = NeuralNet(input_size,hidden_layers,categories).to(device)
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),lr = 0.00005,weight_decay = 0.00001)
J_graph = []
no_of_iter = []
t = 1
for epoch in range(1000):
    for i in range(36):
        image = torch.from_numpy(X_test[i*1000:i*1000+1000,:]).to(device)
        label = torch.from_numpy(Y_test[i*1000:i*1000+1000]).to(device)
        output = model(image.float()).to(device)
        cost = loss(output,label)
        J_graph.append(cost)
        no_of_iter.append(t)
        t = t+1
        if epoch%100==0 and i%36==0:
            print("at epoch {}/1000 cost is {}".format(epoch,cost))
        optim.zero_grad()
        cost.backward()
        optim.step()

plt.plot(J_graph,no_of_iter)
plt.show
with torch.no_grad():
    correct = 0
    samples = 0
    images = torch.from_numpy(X_test).to(device)
    label = torch.from_numpy(Y_test).to(device)
    output = model(images.float()).to(device)
    _,predictions = torch.max(output,1)
    samples += label.shape[0]
    correct += (predictions == label).sum().item()
    print("Train Accuracy: {}".format(correct/samples))
with torch.no_grad():
    correct = 0
    samples = 0
    images = torch.from_numpy(X_vald).to(device)
    label = torch.from_numpy(Y_vald).to(device)
    output = model(images.float()).to(device)
    _,predictions = torch.max(output,1)
    samples += label.shape[0]
    correct += (predictions == label).sum().item()
    print("Test Accuracy: {}".format(correct/samples))
test = pd.read_csv("../input/digit-recognizer/test.csv")
X_test = test.iloc[:,:].values
X_test.shape
X_test = X_test/255
with torch.no_grad():
    images = torch.from_numpy(X_test).to(device)
    label = model(images.float()).to(device)
    _,predictions = torch.max(label,1)
    
predictions.shape
predictions.to('cpu')
labels = np.array(predictions.to('cpu'))
imageid = []
for i in range(X_test.shape[0]):
    imageid.append(i+1)
id = np.array(imageid)
id.shape
labels.shape
submission = pd.DataFrame({"ImageId":imageid,"Label":labels})
submission.to_csv("submission.csv",index = False)
