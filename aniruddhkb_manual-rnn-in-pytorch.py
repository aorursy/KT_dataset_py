import torch
import numpy as np
import matplotlib.pyplot as plt
def gen_data(size):
    x = [np.sin(np.pi*i/32) for i in range(size)]
    y = [ (i > 0).astype(int) for i in x]
    return (x, y)
def initialize_parameters():
    Wy = torch.rand(1)
    by = torch.zeros(1)
    
    Waa = torch.rand(1)
    Wax = torch.rand(1)
    ba = torch.zeros(1)
    
    a = torch.zeros(1)
    
    return Waa, Wax, ba, Wy, by, a
def forward(x, a, Waa, Wax, ba, Wy, by):
    a = torch.tanh(a*Waa + x*Wax + ba)
    yhat = torch.relu(a*Wy + by)
    return yhat, a
    
def onestep(x, y, a, Waa, Wax, ba, Wy, by, costfn, learning_rate):
    x_tens = torch.tensor([x]).float()
    y_tens = y.clone().detach().float()
    Waa = Waa.clone().detach().requires_grad_(True)
    Wax = Wax.clone().detach().requires_grad_(True)
    ba = ba.clone().detach().requires_grad_(True)
    Wy = Wy.clone().detach().requires_grad_(True)
    by = by.clone().detach().requires_grad_(True)
    a = a.clone().detach().requires_grad_(True)
    
    
    yhat, a = forward(x_tens, a, Waa, Wax, ba, Wy, by)
    cost = costfn(yhat, y_tens)
    cost.backward()
    
    with torch.no_grad():
        Waa -= learning_rate*Waa.grad
        Wax -= learning_rate*Wax.grad
        ba -= learning_rate*ba.grad
        Wy -= learning_rate*Wy.grad
        by -= learning_rate*by.grad
    return cost, yhat, a, Waa, Wax, ba, Wy, by
size = 1000000
learning_rate = 0.0001
Waa, Wax, ba, Wy, by, a = initialize_parameters()
costfn = torch.nn.MSELoss()
costs = []
yhats = []
ayes = []
ys = []
y = torch.zeros(1)
for i in range(size):
    x = (np.random.rand(1) > 0.5).astype(int)
    y += x
    ys.append(y.item())
    cost, yhat,a, Waa, Wax, ba, Wy, by = onestep(x, y, a, Waa, Wax, ba, Wy, by, costfn, learning_rate)
    ayes.append(a)
    yhats.append(yhat.item())
    costs.append(cost.item())
    if(i % 10000 == 0):
        print(i, y.item(), yhat.item(), cost.item())
plt.plot(costs)
plt.plot(ys)
plt.plot(yhats)
plt.show()
yhats[-1]
