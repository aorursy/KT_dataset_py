import torch

import torch.nn as nn

import torch.optim as optim

import numpy as np

from fastai.vision import Module

import matplotlib.pyplot as plt

from matplotlib import animation, rc

from IPython.display import HTML
# Defining the model

class LinearRegression(Module):

    def __init__(self, number_of_inputs, number_of_outputs):

        self.linear = nn.Linear(number_of_inputs, number_of_outputs)

        

    def forward(self, x):

        return self.linear(x) 
# Defining the fit function

def fit(inputs, targets, model, criterion, optimizer, num_epochs):

    loss_history = [] # to save the loss at each epoch.

    

    for epoch in range(num_epochs):

        # forward pass

        out = model(inputs)          

        loss = criterion(out, targets) 



        # backward pass

        optimizer.zero_grad() 

        loss.backward()

        optimizer.step()

        

        # store value of loss

        loss_history.append(loss.item())

        

    print('Epoch[{}/{}], loss:{:.6f}'.format(epoch+1, num_epochs, loss.item()))

    return loss_history
# Model, Criterion and Optimizer

model = LinearRegression(1,1)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.1) 
# Create data

x_train = torch.linspace(0, 1, 10000) 

y_train = 2*x_train + 1 + torch.randn(x_train.size())*x_train*0.1 

x_train = x_train.unsqueeze(-1)

y_train = y_train.unsqueeze(-1)
# Visualize our data

plt.figure(figsize=(6,3), dpi=120)

plt.scatter(x_train.numpy(), y_train.numpy(), color='gray', label='Noisy data (model inputs)') 

plt.plot(x_train.numpy(), (2*x_train + 1).numpy(), color='black', label='True data')

plt.xlabel('x')

plt.ylabel('y = 2x + 1 + noise')

plt.legend()

plt.show();

print(x_train.size(), y_train.size()) 
%time # Measure the time of execussion

loss = fit(x_train.requires_grad_(True), 

           y_train, 

           model, 

           criterion, 

           optimizer, 

           num_epochs=200)
plt.figure(figsize=(6,3), dpi=120)

plt.plot(loss, color='gray')

plt.xlabel('epoch')

plt.ylabel('Loss')

plt.show();
list(model.parameters())
# Compute model estimates

model.eval()        # set model to evaluation mode

ye = model(x_train) # compute the y estimate
# Visualize our data

plt.figure(figsize=(6,3), dpi=120)

plt.scatter(x_train.detach().numpy(), y_train.numpy(), color='gray', label='Noisy data (model inputs)') 

plt.plot(x_train.detach().numpy(), (2*x_train + 1).detach().numpy(), color='black', label='True data')

plt.plot(x_train.detach().numpy(), ye.detach().numpy(), color='blue', label='Model estimate')

plt.xlabel('x')

plt.ylabel('y = 2x + 1 + noise')

plt.legend()

plt.show();
x_true = torch.linspace(-2, 2, 1000)

y_true = 3*x_true**2 + 2*x_true + 1

y_train =  y_true + torch.randn(x_true.size())



x_true.unsqueeze_(-1)

y_train.unsqueeze_(-1)



x_train = torch.cat((x_true**2, x_true), dim=1) # Concatenate x and x**2 to a N x 2 x_train matrix.

print(x_train.size(), y_train.size()) 
# Visualize our data

plt.figure(figsize=(6,3), dpi=120)

plt.scatter(x_true.numpy(), y_train.numpy(), color='gray', label='Observed data') 

plt.plot(x_true.numpy(), y_true.numpy(), color='black', label='True data')

plt.xlabel('x')

plt.ylabel(r'$y = 3x^2 + 2x + 1 + noise$') # Notice you can use latex in the label string

plt.legend()

plt.show();
model = LinearRegression(2, 1)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.1) 
%time 

loss = fit(x_train, y_train, model, criterion, optimizer, num_epochs=250)
list(model.parameters())
model.eval()

ye = model(x_train) # compute the y estimate

ye = ye.detach().numpy() # get the values from the variable, them pass them to the cpu and convert to a numpy array
# Visualize our data

plt.figure(figsize=(6,3), dpi=120)

plt.scatter(x_true.numpy(), y_train.numpy(), color='gray', label='Observed data') 

plt.plot(x_true.numpy(), y_true.numpy(), color='black', label='True data')

plt.plot(x_true.numpy(), ye, color='blue', label='Predicted data')

plt.xlabel('x')

plt.ylabel(r'$y = 3x^2 + 2x + 1 + \epsilon$') # Notice you can use latex in the label string

plt.legend()

plt.show();
def fit2(inputs, targets, model, criterion, optimizer, num_epochs):

    loss_history = [] # to save the loss at each epoch.

    out_history = [] # to save the parameters at each epoch

    for ii, epoch in enumerate(range(num_epochs)):

        # forward

        out = model(inputs)

        loss = criterion(out, targets)



        # backward

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        loss_history.append(loss.item())

        

        if ii == 0:

            out_history = out.detach().numpy()

        else:

            out_history = np.concatenate((out_history, out.detach().numpy()), axis=-1)

        

    print('Epoch[{}/{}], loss:{:.6f}'.format(epoch+1, num_epochs, loss.item()))

    return loss_history, out_history
model = LinearRegression(2, 1)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.1) 
%time 

loss, out = fit2(x_train, y_train, model, criterion, optimizer, num_epochs=250)
%%capture

# First set up the figure, the axis, and the plot element we want to animate

fig, ax = plt.subplots(figsize=(6,3), dpi=120)

ax.set_xlim((-2.2, 2.2))

ax.set_ylim((-5, 20))

ax.plot(x_true.numpy(), y_true.numpy(), lw=2, color='black', label='True model')

ax.scatter(x_true.numpy(), y_train.numpy(), color='gray', label='Observed data') 

ax.set_ylabel(r'$y = 3x^2 + 2x + 1 + noise$') # Notice you can use latex in the label string

line, = ax.plot([], [], lw=2, label='Predicted model')

ax.legend()



# animation function. This is called sequentially

def animate(i):

    line.set_data(x_true.numpy(), out[...,i])

    return (line,)



# call the animator. blit=True means only re-draw the parts that have changed.

anim = animation.FuncAnimation(fig, animate, frames=out.shape[1], interval=30, blit=True)
HTML(anim.to_html5_video())
x_train = torch.linspace(-2, 2, 1000)

x_train.unsqueeze_(-1)

y_true = 6*x_train**3 + 3*x_train**2 + 2*x_train + 1

y_train =  y_true + torch.randn(x_train.size())*5



print(x_train.size(), y_train.size()) 
# Visualize our data

plt.figure(figsize=(6,3), dpi=120)

plt.scatter(x_train.numpy(), y_train.numpy(), color='gray', label='Observed data') 

plt.plot(x_train.numpy(), y_true.numpy(), color='black', label='True model')

plt.xlabel('time')

plt.ylabel('y') # Notice you can use latex in the label string

plt.legend()

plt.show();
class GeneralFit(Module):

    def __init__(self, input_size, output_size, hidden_size=100):

        self.linear_in  = nn.Linear(input_size, hidden_size)

        self.hidden     = nn.Linear(hidden_size, hidden_size)

        self.linear_out = nn.Linear(hidden_size, output_size)

        

    def forward(self, x):

        x = torch.relu(self.linear_in(x))

        x = torch.relu(self.hidden(x))

        x = self.linear_out(x)

        return x
model = GeneralFit(1, 1)

criterion = nn.MSELoss() # Tentar também L1loss

optimizer = optim.Adam(model.parameters(), lr=0.01)
%time 

loss, out = fit2(x_train.requires_grad_(True),

                y_train, 

                model, 

                criterion, 

                optimizer, 

                num_epochs=200)
plt.figure(figsize=(6,3), dpi=120)

plt.plot(loss, color='gray')

plt.xlabel('epoch')

plt.ylabel('Loss')

plt.show();
ye = model(x_train) # compute the y estimate

ye = ye.detach().numpy() # get the values from the variable, them pass them to the cpu and convert to a numpy array
# Visualize our data

plt.figure(figsize=(6,3), dpi=120)

plt.scatter(x_train.detach().numpy(), y_train.numpy(), color='gray', label='Observed data') 

plt.plot(x_train.detach().numpy(), y_true.numpy(), color='black', label='True model')

plt.plot(x_train.detach().numpy(), ye, color='blue', label='Predicted model')

plt.xlabel('time')

plt.ylabel('y')

plt.legend()

plt.show();
%%capture

# First set up the figure, the axis, and the plot element we want to animate

fig, ax = plt.subplots(figsize=(6,3), dpi=120)

#ax.set_xlim((-2.2, 2.2))

#ax.set_ylim((-5, 20))

ax.plot(x_train.detach().numpy(), y_true.numpy(), lw=2, color='black', label='True model')

ax.scatter(x_train.detach().numpy(), y_train.numpy(), color='gray', label='Observed data') 

ax.set_ylabel('y') # Notice you can use latex in the label string

line, = ax.plot([], [], lw=2, label='Predicted model')

ax.legend()



# animation function. This is called sequentially

def animate(i):

    line.set_data(x_train.detach().numpy(), out[...,i])

    return (line,)



# call the animator. blit=True means only re-draw the parts that have changed.

anim = animation.FuncAnimation(fig, animate, frames=out.shape[1], interval=30, blit=True)
HTML(anim.to_html5_video())