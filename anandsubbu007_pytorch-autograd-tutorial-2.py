#Importing Libraries
import torch
# Assign Device 
cuda0 = torch.device("cuda:0")
x = torch.tensor([5],dtype=torch.float32,requires_grad=True)
y = torch.tensor([6],dtype=torch.float32,requires_grad=True)
print(x)
print(y)
#defining the function
z = ((x**2)*y) + (x*y)
print(z)
#Using autograd
# Autograd to be applied on Scalars
total = torch.sum(z) # Converting to scalar
total
print(x.grad,y.grad)
total.backward() # to call grad function we need to call .backward() if not it will show as None
print("Def with resp. to  x   :",x.grad)
print("Def with resp. to  y   :",y.grad)
x = torch.randint(-100,100,(100,), dtype = torch.float32 , device = cuda0)
y = (1.32*x) + 25                       # y = (w*x) + b     we are going to predict w & b
w = torch.ones(1,requires_grad = True, device = cuda0 )
b = torch.ones(1,requires_grad = True, device = cuda0 ) 
y_hat = (w*x) + b

epochs = 10000
lr = 0.000001
count = 0
for i in range(epochs):
  loss = torch.sum((y_hat - y)**2) 
  loss.backward() 
  #w -= lr*w.grad --> this will be considered as relationship
  with torch.no_grad(): # this will switch off gradients

    w -= lr*w.grad
    b -= lr*b.grad
    count += 1
    #setting gradients to be zero
    w.grad.zero_()
    b.grad.zero_() 
  
  y_hat = (w*x ) + b

print(count)
print("Predicted w value  :",w.item())
print("Predicted b value  :",b.item())