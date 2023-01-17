import numpy as np
# f = w*x --> f = 2*x

X = np.array([1,2,3,4],dtype=np.float32)

y = np.array([2,4,6,8], dtype=np.float32)



w = 0.0



# model prediction

def forward(x):

    return w*x



# loss = MSE

def loss(y,y_predicted):

    return ((y_predicted-y)**2).mean()





# gradient

# MSE = 1/N * (w*x - y)**2

# dJ/dw = 1/N 2x (w*x - y)

def gradient(x,y,y_predicted):

    return np.dot(2*x, (y_predicted-y)).mean()





print(f'Prediction before training: f(5) = {forward(5):.3f}')



# Training

learning_rate = 0.01

n_iters = 10



for epoch in range(n_iters):

    # prediction = forward pass

    y_pred = forward(X)

    

    # loss

    l = loss(y,y_pred)

    

    # gradients

    dw = gradient(X,y,y_pred)

    

    # update weights

    w -= learning_rate * dw

    

    if epoch % 1==0:

        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

        





print(f'Prediction after training: f(5) = {forward(5):.3f}')
import torch

# f = w*x --> f = 2*x

X = torch.tensor([1,2,3,4],dtype=torch.float32)

y = torch.tensor([2,4,6,8], dtype=torch.float32)



w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)



# model prediction

def forward(x):

    return w*x



# loss = MSE

def loss(y,y_predicted):

    return ((y_predicted-y)**2).mean()





print(f'Prediction before training: f(5) = {forward(5):.3f}')



# Training

learning_rate = 0.01

n_iters = 10



for epoch in range(n_iters):

    # prediction = forward pass

    y_pred = forward(X)

    

    # loss

    l = loss(y,y_pred)

    

    # gradients = backward pass

    l.backward() # dl/dw

    

    # update weights

    with torch.no_grad():

        w -= learning_rate * w.grad

    

    # zero gradients

    w.grad.zero_()

    if epoch % 1==0:

        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

        





print(f'Prediction after training: f(5) = {forward(5):.3f}')
import torch.nn as nn
import torch

# f = w*x --> f = 2*x

X = torch.tensor([1,2,3,4],dtype=torch.float32)

y = torch.tensor([2,4,6,8], dtype=torch.float32)



w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)



# model prediction

def forward(x):

    return w*x



print(f'Prediction before training: f(5) = {forward(5):.3f}')



# Training

learning_rate = 0.01

n_iters = 10



loss = nn.MSELoss()

optimizer = torch.optim.SGD([w], lr = learning_rate)



for epoch in range(n_iters):

    # prediction = forward pass

    y_pred = forward(X)

    

    # loss

    l = loss(y,y_pred)

    

    # gradients = backward pass

    l.backward() # dl/dw

    

    # update weights

    optimizer.step()

    

    # zero gradients

    optimizer.zero_grad()

    

    if epoch % 1==0:

        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

        





print(f'Prediction after training: f(5) = {forward(5):.3f}')
import torch

# f = w*x --> f = 2*x



# Input size have a certain size

X = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)

y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)



X_test = torch.tensor([5], dtype= torch.float32)

n_samples, n_features = X.shape

print(n_samples, n_features)



input_size = n_features

output_size = n_features



model = nn.Linear(input_size, output_size)



print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')



# Training

learning_rate = 0.01

n_iters = 10



loss = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)



for epoch in range(n_iters):

    # prediction = forward pass

    y_pred = model(X)

    

    # loss

    l = loss(y,y_pred)

    

    # gradients = backward pass

    l.backward() # dl/dw

    

    # update weights

    optimizer.step()

    

    # zero gradients

    optimizer.zero_grad()

    

    if epoch % 1==0:

        [w,b] = model.parameters()

        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

        





print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
import torch

# f = w*x --> f = 2*x



# Input size have a certain size

X = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)

y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)



X_test = torch.tensor([5], dtype= torch.float32)

n_samples, n_features = X.shape

print(n_samples, n_features)



input_size = n_features

output_size = n_features



model = nn.Linear(input_size, output_size)



class LinearRegression(nn.Module):

    

    def __init__(self,input_dim,output_dim):

        super(LinearRegression,self).__init__()

        

        self.lin = nn.Linear(input_dim,output_dim)

    def forward(self,x):

        return self.lin(x)



model = LinearRegression(input_size, output_size)



print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')



# Training

learning_rate = 0.01

n_iters = 10



loss = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)



for epoch in range(n_iters):

    # prediction = forward pass

    y_pred = model(X)

    

    # loss

    l = loss(y,y_pred)

    

    # gradients = backward pass

    l.backward() # dl/dw

    

    # update weights

    optimizer.step()

    

    # zero gradients

    optimizer.zero_grad()

    

    if epoch % 1==0:

        [w,b] = model.parameters()

        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

        





print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')