import warnings

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))



import torch

from torch.autograd import Variable

import numpy as np



xy = np.loadtxt("../input/diabetes.csv", delimiter=',', dtype=np.float32)

y_data = Variable(torch.from_numpy(xy[:,-1]))
import pandas as pd

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



data = pd.DataFrame(xy)

data.head(5)
data = np.array(data)

data = sc.fit_transform(data)

data = pd.DataFrame(data)

data.head(5)
xy = np.array(data)
x_data = Variable(torch.from_numpy(xy[:,:-1]))



print(x_data.data.shape)

print(y_data.data.shape)
class Model(torch.nn.Module):

    

    def __init__(self):

        super(Model,self).__init__()

        self.l1 = torch.nn.Linear(8,6)

        self.l2 = torch.nn.Linear(6,4)

        self.l3 = torch.nn.Linear(4,2)

        self.l4 = torch.nn.Linear(2,1)

        

        self.sigmoid = torch.nn.Sigmoid()

    

    def forward(self, x):

        out1 = self.sigmoid(self.l1(x))

        out2 = self.sigmoid(self.l2(out1))

        out3 = self.sigmoid(self.l3(out2))

        y_pred = self.sigmoid(self.l4(out3))

        return y_pred

    

model = Model()
criterion = torch.nn.BCELoss(size_average=True)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(1000):

    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)

    

    if epoch%100 == 0:

        print(epoch, loss.item())

        

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()