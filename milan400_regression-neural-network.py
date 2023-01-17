import torch

from torch.autograd import Variable

import torch.nn.functional as F

import torch.utils.data as Data



import matplotlib.pyplot as plt

%matplotlib inline



import numpy as np

import imageio
#reproducible

torch.manual_seed(1)



# x data (tensor), shape=(100, 1)

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)

print(type(x))



 # noisy y data (tensor), shape=(100, 1)

y = x.pow(2) + 0.2*torch.rand(x.size())



#torch can only train on Variable, so convert them to Variable

x, y = Variable(x),Variable(y)



#view data

plt.figure(figsize=(10,4))

plt.scatter(x.data.numpy(), y.data.numpy(),color="orange")

plt.title('Regression Analysis')

plt.xlabel('Independent Variable')

plt.ylabel('Dependent Variable')

plt.show()
class Net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):

        super(Net, self).__init__()

        #hidden layer

        self.hidden = torch.nn.Linear(n_feature,n_hidden)

        #output layer

        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):

        #activation function for hidden layer

        x = F.relu(self.hidden(x))

        #linear output

        x = self.predict(x)

        return(x)
#define the network

net = Net(n_feature=1, n_hidden=10, n_output=1)

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)

#Regression Mean Squared loss

loss_func = torch.nn.MSELoss()
my_images = []

fig, ax = plt.subplots(figsize=(12,7))



#train the network

for t in range(200):

    #input x and prediction based on x

    prediction = net(x)

    

    loss = loss_func(prediction, y)

    

    #clear gradients for next train

    optimizer.zero_grad()

    

    #backpropagation, compute gradients

    loss.backward()

    

    #apply gradient

    optimizer.step()

    

    #plot and show learning process

    plt.cla()

    

    ax.set_title('Regression Analysis',fontsize=35)

    ax.set_xlabel('Independent variable',fontsize=24)

    ax.set_ylabel('Dependent variable',fontsize=24)

    

    ax.set_xlim(-1.05,1.5)

    ax.set_ylim(-0.25, 1.25)

    ax.scatter(x.data.numpy(), y.data.numpy(), color="orange")

    ax.plot(x.data.numpy(), prediction.data.numpy(), 'g-', lw=3)

    ax.text(1.0, 0.1, 'Step = %d' % t, fontdict={'size': 24, 'color':  'red'})

    ax.text(1.0, 0, 'Loss = %.4f' % loss.data.numpy(),fontdict={'size': 24, 'color':  'red'})

    

    #draw the canvas, cache the renderer

    fig.canvas.draw()

    image = np.frombuffer(fig.canvas.tostring_rgb(),dtype='uint8')

    image = image.reshape(fig.canvas.get_width_height()[::-1]+(3,))

    

    my_images.append(image)



#save images as a gif

imageio.mimsave('./curve_1.gif', my_images, fps=10)

    