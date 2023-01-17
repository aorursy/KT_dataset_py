import torch
x = torch.tensor(2.5)
m = torch.tensor(0.,requires_grad=True)
b = torch.tensor(0.,requires_grad=True)
y_hat = m*x+b
y = 5.
loss = (y-y_hat)**2
loss.backward()
print ("m grad:",m.grad)
print ("b grad:",b.grad)
lr = 0.01
m = m-lr*m.grad
b = b-lr*b.grad
print ("m:",m)
print ("b:",b)
y_hat = m*x+b
y = 5.
loss = (y-y_hat)**2
print ("New loss:",loss)
