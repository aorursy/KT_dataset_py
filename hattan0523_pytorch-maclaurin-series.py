import torch



x = torch.tensor([3.], requires_grad=True)

f = x ** 3

f.backward()

print(x.grad)
x = torch.tensor([3.], requires_grad=True)

f = x ** 3

g = torch.autograd.grad(f, x, create_graph=True)

g[0].backward()

x.grad


x = torch.tensor([3.], requires_grad=True)

f = x ** 3

g = torch.autograd.grad(f, x, create_graph=True)

h = torch.autograd.grad(g, x, create_graph=True)

i = torch.autograd.grad(h, x, create_graph=True)

i[0].backward()

print(g)

print(h)

print(i)

print(x.grad)
import torch

coefficients_order=10



def grad(f, x):

    return torch.autograd.grad(f, x, create_graph=True)



x = torch.tensor([0.], requires_grad=True)

f = torch.sin(x)

result = []

grad_result = [grad(f, x)]

for i in range(coefficients_order):

    grad_result.append(grad(grad_result[-1], x))    



cnt = 1.

for num, data in enumerate(grad_result):

    result.append(data[0].detach()/cnt)

    cnt = cnt * (num + 2)

    

coefficient = torch.stack(result)
x = torch.linspace(0, 10, 100)

matrix = torch.zeros(x.shape[0], coefficient.shape[0])

for i in range(coefficient.shape[0]):

    matrix[:,i] = x **(i+1)

matrix.shape
result_y = torch.mm(matrix, coefficient)
import matplotlib.pyplot as plt

import numpy as np

plt.plot(x.detach().numpy()[:55], result_y.detach().numpy()[:55], label="pytorch \nMaclaurin series \norder={}".format(coefficients_order))

plt.plot(x.detach().numpy()[:55], np.sin(x.detach().numpy()[:55]), label="sin(x)")

plt.legend(loc="best")

plt.xlabel("x")

plt.ylabel("y")

plt.show()
x