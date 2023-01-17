import torch

from torch.autograd import Variable

from torch import nn

from torch import optim



import numpy as np

import matplotlib.pyplot as plt
#graphical functions



def plot_points(X, y, m="_"):

    class0 = X[np.argwhere(y==0)]

    class1 = X[np.argwhere(y==1)]

    class2 = X[np.argwhere(y==2)]

    plt.scatter([s[0][0] for s in class0], [s[0][1] for s in class0], s = 25, color = 'red', edgecolor = 'k',zorder=2, marker=m)

    plt.scatter([s[0][0] for s in class1], [s[0][1] for s in class1], s = 25, color = 'blue', edgecolor = 'k',zorder=2, marker=m)

    plt.scatter([s[0][0] for s in class2], [s[0][1] for s in class2], s = 25, color = 'yellow', edgecolor = 'k',zorder=2, marker=m)



def plot_zone(X, y):

    class0 = X[np.argwhere(y==0)]

    class1 = X[np.argwhere(y==1)]

    class2 = X[np.argwhere(y==2)]    

    plt.scatter([s[0][0] for s in class0], [s[0][1] for s in class0], s = 25, color = 'red', edgecolor = 'k',zorder=2, marker="+")

    plt.scatter([s[0][0] for s in class1], [s[0][1] for s in class1], s = 25, color = 'blue', edgecolor = 'k',zorder=2, marker="_" )  

    plt.scatter([s[0][0] for s in class2], [s[0][1] for s in class2], s = 25, color = 'yellow', edgecolor = 'k',zorder=2, marker="_" )  

    



def view(net):



    datamin = np.min(net.points, axis=0)

    datamax = np.max(net.points, axis=0)

    xmin, ymin = datamin

    xmax, ymax = datamax    

       

    border = []

    xList = np.linspace(xmin, xmax, 50)

    for x in xList:

        yList = np.linspace(ymin, ymax, 50)

        for y in yList:

            out = net.forward(torch.Tensor([[x,y]]))

            #because net has logsoftmax at forward

            #get back probabilities by exp

            out = torch.exp(out)

            if(out[0][0].data.numpy()>0.5):    

                border.append([x,y,0])

            if(out[0][1].data.numpy()>0.5):    

                border.append([x,y,1])

            if(out[0][2].data.numpy()>0.5):    

                border.append([x,y,2])

                

    border = np.array(border)

    solutionX = border[:, [0,1]]

    solutionT = border[:, [2]]

    plot_zone(solutionX, solutionT)



    plot_points(net.points, net.target,m="o")  
#define data



#from array

import pandas as pd

data = np.array([

    [1,10,0],

    [3,10,1],

    [1.8,2.0,2],

    [0,6,0],

    [-1,-1,1],

    [-2,10,2],

])

#or

#from file

data = pd.read_csv('../input/data1.csv', header=None)

data = data.reset_index().values #.to_numpy()

data = data[:,[1,2,3]]

#print(data)
class NN(nn.Module):

    def __init__(self):

        torch.manual_seed(1)

        

        super().__init__()

        

        self.fc1 = nn.Linear(2, 10)

        self.fc2 = nn.Linear(10, 10)        

        self.output = nn.Linear(10, 3)

        

        self.relu = nn.ReLU()

        #also could use sigmoid instead of ReLU

        #self.sigmoid = nn.Sigmoid()

        self.logsoftmax = nn.LogSoftmax(dim=1)       



    def forward(self, x):

        # Pass the input tensor through each of our operations

        x = self.fc1(x)

        x = self.relu(x)  

        x = self.fc2(x)

        x = self.relu(x)

        x = self.output(x)

        x = self.logsoftmax(x)

        

        return x        



def train(self, data, epochs = 100, alpha=0.01):

    

    self.epochs = epochs

    self.alpha = alpha        



    self.points = data[:, [0,1]]

    self.target = data[:, [2]]

    self.T = torch.from_numpy(self.target)

    self.T = self.T.t()[0].long()

    self.features = torch.from_numpy(net.points).float()   

    

    self.lossHistory = []



    # Optimizers require the parameters to optimize and a learning rate

    optimizer = optim.SGD(self.parameters(), lr=self.alpha)    

    

    for i in range(epochs):

        output = self.forward(self.features)



        loss = nn.NLLLoss()(output,self.T)

        

        self.lossHistory.append(loss)



        # clear previous gradients, otherwise they will be accumulated

        optimizer.zero_grad()        



        # calculate gradients

        loss.backward()



        # update weights

        optimizer.step()
net = NN()

net.train = train

net.train(net,data,epochs=2000,alpha=0.01)





print(net.lossHistory[len(net.lossHistory)-1])

plt.plot(net.lossHistory)

plt.show()



view(net)