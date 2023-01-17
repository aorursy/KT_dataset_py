def compute_gradient(x):
    # The framework like Tensorflow and Pytorch
    # have done things for us.
    pass
# SGD
while True:
    dx = compute_gradient(x)
    x += learning_rate * dx
    
# SGD + Momentum
vx = 0
while True:
    dx = compute_gradient(x)
    vx = rho * vx + dx
    x += learning_rate * vx
    
# AdaGrad
grad_squared = 0
while True:
    dx = compute_gradient(x)
    grad_squared += dx * dx
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
    
# RMSprop
grad_squared = 0
while True:
    dx = compute_gradient(x)
    grad_squared += decay_rate * grad_squared + (1 - decay_rate) * dx * dx
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)

first_moment = 0
second_moment = 0
while True:
    dx = compute_gradient(x)
    # here comes the method of SGD momentum
    first_moment = B1 * first_moment + (1 - B1) * dx
    # here comes the method of AdaGrad/RMSProp
    second_moment = B2 * second_moment + (1 - B2) * dx * dx
    first_unbias = first_moment / (1 - B1 ** t)
    second_unbias = second_moment / (1 - B2 ** t)
    # two of these codes above are used to prevent the beginning stride from too big value
    x -= learning_rate * first_moment / ((np.sqrt(second_moment) + 1e-7))

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as fn

import matplotlib.pyplot as plt
# Hyperparameters
learning_rate = 0.01
batch_size = 32
epochs = 12
# Create Data
x = torch.linspace(-1, 1, 1000).unsqueeze(-1)
y = x.pow(2) + 0.05 * torch.normal(torch.zeros(*x.size()))
# Wrap data into dataset
dataset = data.TensorDataset(x, y)
loader = data.DataLoader(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=2)
# Construct a model
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
net_SGD = FC()
net_Momentum = FC()
net_RMSprop = FC()
net_Adam = FC()
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

# different optimizers
opt_SGD = torch.optim.SGD(net_SGD.parameters(), 
                          lr=learning_rate)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), 
                               lr=learning_rate, 
                               momentum=0.8)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), 
                                  lr=learning_rate, 
                                  alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), 
                            lr=learning_rate, 
                            betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()
losses_his = [[], [], [], []]   # record loss

# training
for epoch in range(epochs):
    print('Epoch: ', epoch)
    for step, (b_x, b_y) in enumerate(loader):          # for each training step
        for net, opt, l_his in zip(nets, optimizers, losses_his):
            output = net(b_x)              # get output for every net
            loss = loss_func(output, b_y)  # compute loss for every net
            opt.zero_grad()                # clear gradients for next train
            loss.backward()                # backpropagation, compute gradients
            opt.step()                     # apply gradients
            l_his.append(loss.data.numpy())     # loss recoder

labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()

