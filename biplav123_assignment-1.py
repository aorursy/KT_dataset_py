# Import torch and other required modules
import torch
import numpy as np
x = torch.ones(1, requires_grad=True)
y = x**2

z = x**3

print("x = ", x)
print("y = ", y)
print("z = ", z)
y.backward(retain_graph=True) # (retain_graph=True)
print("dy/dx = ", x.grad)

x.grad.data.zero_()

z.backward(retain_graph=True) # (retain_graph=True)
print("dy/dx = ", x.grad)
y.backward(retain_graph=True) # (retain_graph=True)
print("dy/dx = ", x.grad)

# x.grad.data.zero_()

z.backward(retain_graph=True) # (retain_graph=True)
print("dy/dx = ", x.grad)
w = np.array([[2., 2.],[2., 2.]])
x = np.array([[3., 3.],[3., 3.]])
b = np.array([[4., 4.],[4., 4.]])
w = torch.tensor(w, requires_grad=True)
x = torch.tensor(x, requires_grad=True)
b = torch.tensor(b, requires_grad=True)


y = w*x + b 
print(y)
# tensor([[10., 10.],
#         [10., 10.]], dtype=torch.float64, grad_fn=<AddBackward0>)

y.backward(torch.FloatTensor([[1, 1],[ 1, 1]]))

print(w.grad)
# tensor([[3., 3.],
#         [3., 3.]], dtype=torch.float64)

print(x.grad)
# tensor([[2., 2.],
#         [2., 2.]], dtype=torch.float64)

print(b.grad)
# tensor([[1., 1.],
#         [1., 1.]], dtype=torch.float64)
w = np.array([[2., 2.],[2., 2.]])
x = np.array([[3., 3.],[3., 3.]])
b = np.array([[4., 4.],[4., 4.]])
w = torch.tensor(w, requires_grad=True)
x = torch.tensor(x, requires_grad=True)
b = torch.tensor(b, requires_grad=True)


y = w*x + b 
print(y)
y.backward(torch.FloatTensor([[1, 1],[ 1, 1]]), retain_graph = True)

print("dy/dx = ", x.grad) # w
print("dy/dw = ", w.grad) # x
print("dy/db = ", b.grad) # 1
w = np.array([[2., 2.],[2., 2.]])
x = np.array([[3., 3.],[3., 3.]])
b = np.array([[4., 4.],[4., 4.]])
w = torch.tensor(w, requires_grad=True)
x = torch.tensor(x, requires_grad=True)
b = torch.tensor(b, requires_grad=True)


y = w*x + b 
y.backward()
import numpy as np
x = np.array([1., 4., 3., 2., 5., 10., 8.])
y = np.array([[3.,4.], [1.,2.], [7., 8.]])
x = torch.tensor(x)
y = torch.tensor(y)
print("x = \n", x)
print("y = \n", y)
print(x.sort())
print("y sorted along rows\n",torch.sort(y))
print("y sorted along columns\n",torch.sort(y, dim = 0))

aa = np.linspace(1, 10, num=10)
aa = torch.tensor(aa)
print(aa)
print(aa.reshape(2,5)) # 2 rows 5 columns
print(aa.reshape(5,2)) # 5 rows 2 columns
print(aa.reshape(5,-1)) # 2 rows 5 columns
print(torch.rand(4,5))
aa = torch.rand(4,4)
print (aa)
bb = torch.bernoulli(aa)
print(bb)

bb
aa = torch.ones(4,4)
print (aa)
bb = torch.bernoulli(aa)
print(bb)
!pip install jovian --upgrade --quiet
import jovian
jovian.commit()
