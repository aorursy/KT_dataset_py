import torch

import torch.nn as nn

import torch.nn.functional as fn

import matplotlib.pyplot as plt
data = torch.linspace(-1,1,100)

x = torch.unsqueeze(data,dim=1)

y = - x.pow(2) + 0.5 * torch.rand(x.size())
class regression(nn.Module):

    def __init__(self,n_input,n_hidden,n_output):

        super(regression,self).__init__()

        self.fc1 = nn.Linear(n_input,n_hidden)

        self.fc2 = nn.Linear(n_hidden,n_hidden)

        self.fc3 = nn.Linear(n_hidden,n_output)

        self.relu = nn.ReLU()

    

    def forward(self,x):

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        x = self.relu(x)

        x = self.fc3(x)

        return x
reg_net = regression(1,20,1)

optimizer = torch.optim.SGD(reg_net.parameters(),lr = 0.2)

loss = torch.nn.MSELoss()
plt.ion()

for i in range(200):

    pred = reg_net(x)

    

    loss_value = loss(pred,y)

    

    optimizer.zero_grad()

    loss_value.backward()

    optimizer.step()

    

    if i % 5 == 0:

        plt.cla()

        plt.scatter(x.data.numpy(),y.data.numpy())

        plt.plot(x.data.numpy(),pred.data.numpy(),'g-',lw=5)

        plt.text(0.5,0,'Loss = {:.4}'.format(loss_value.data.numpy()),fontdict={'size':20,'color':'red'})

        plt.pause(0.1)

    plt.ioff()

    plt.show()