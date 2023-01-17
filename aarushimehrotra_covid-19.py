# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# load data into numpy array
raw = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

# separate data by country
cnames = set()
snames = set()
data = dict()
for i in raw.to_numpy():
    if type(i[1]) == float:
        cnames.add(i[2])
    else:
        snames.add(i[1])
        
for cname in cnames:
    current = raw[raw["Country_Region"] == cname].to_numpy()
    data.update({cname: current})
for sname in snames:
    current = raw[raw["Province_State"] == sname].to_numpy()
    data.update({sname: current})
    
# x and y values
x_data = list(range(len(data[list(data.keys())[0]])))
y_data = dict()
for (name, cases) in data.items():
    y_data.update({name : cases[:,4]})
y_data
from torch.utils.data import Dataset, DataLoader

name = "Zimbabwe"

class CovidDataset(Dataset):
    def __init__(self):
        self.x = torch.FloatTensor([[float(i)] for i in x_data])
        self.y = torch.FloatTensor([[i/1.8e4] for i in y_data[name]])
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)
    
dataset = CovidDataset()
loader = DataLoader(dataset=dataset, batch_size=70, shuffle=True, num_workers=4)
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(1, 4)
        self.sig = torch.nn.Sigmoid()
        self.l2 = torch.nn.Linear(4, 2)
        self.sig2 = torch.nn.Sigmoid()
        self.l3 = torch.nn.Linear(2, 1)
    
    def forward(self, x):
        out1 = self.sig(self.l1(x/70))
        out2 = self.sig2(self.l2(out1))
        y_pred = self.l3(out2)
        return y_pred

model = Model()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())


for epoch in range(10000):
    for i, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        y_pred = model(inputs)

        loss = criterion(y_pred, labels)
        print(epoch, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

x_pred = [[float(i)] for i in range(150)]
y_pred = [i.item()*1.8e4 for i in model(torch.FloatTensor(x_pred)).detach().numpy()]
plt.plot(x_pred, y_pred, label="model")
plt.plot(x_data, y_data[name], label="actual")
plt.legend()
plt.show()
