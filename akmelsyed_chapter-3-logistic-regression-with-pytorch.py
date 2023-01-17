import torch

import matplotlib.pyplot as plt

from IPython import display

import time
X = torch.cat((torch.arange(15, dtype=float), torch.arange(20, 35, dtype=float)), axis=0)

y = torch.cat((torch.zeros(15, dtype=float), torch.ones(15, dtype=float)), axis=0)
print('X')

print(X)

print('y')

print(y)
def log_loss(y, y_pred): ##log loss error / binary cross entropy

  return -torch.sum((y*torch.log(y_pred) + (1-y)*torch.log(1-y_pred)))/y.shape[0]
epoch_loss = []



weights = torch.tensor([0., 0.])

learning_rate = 1e-4

n = X.shape[0]



for epoch in range(600000+1): 

    linear = weights[0]*X + weights[1]

    y_pred = 1/(1+torch.exp(-linear)) ##logistic

    loss = log_loss(y, y_pred)

    epoch_loss.append(loss)





    if(epoch%50000 == 0):

        ######plotting#####

        display.display(plt.gcf())

        display.clear_output(wait=True)

        fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)

        fig.suptitle('epoch = {0}'.format(epoch))

        ax0.scatter(X, y)

        ax0.plot(X, y_pred, 'r')

        ax0.set_title('slope = {0:.5f}, bias = {1:.5f}'.format(weights[0], weights[1]))

        ax1.set_title('loss = {0:.2f}'.format(loss))

        ax1.plot(epoch_loss)

        plt.show()

        time.sleep(1)

        ###################

    

    ###simplified slope and bias derivatives with respect to loss###  

    D_mse_wrt_slope = torch.sum(X * (y - y_pred))

    D_mse_wrt_bias = torch.sum(y - y_pred)

    ###########



    weights[0] += learning_rate * D_mse_wrt_slope

    weights[1] += learning_rate * D_mse_wrt_bias
epoch_loss = []



weights = torch.tensor([0., 0.], requires_grad=True)

learning_rate = 1e-2

n = X.shape[0]



for epoch in range(60000+1):

    linear = weights[0]*X + weights[1]

    y_pred = 1/(1+torch.exp(-linear)) ##logistic

    loss = log_loss(y, y_pred)

    epoch_loss.append(loss.item())





    if(epoch%5000 == 0):

        ######plotting#####

        display.display(plt.gcf())

        display.clear_output(wait=True)

        fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)

        fig.suptitle('epoch = {0}'.format(epoch))

        ax0.scatter(X, y)

        ax0.plot(X, y_pred.detach().numpy(), 'r')

        ax0.set_title('slope = {0:.5f}, bias = {1:.5f}'.format(weights[0], weights[1]))

        ax1.set_title('loss = {0:.2f}'.format(loss))

        ax1.plot(epoch_loss)

        plt.show()

        time.sleep(1)

        ###################



    ###slope and bias derivatives with respect to loss###

    loss.backward()



    with torch.no_grad():

        weights -= learning_rate * weights.grad

        

    weights.grad.zero_()