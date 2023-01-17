# Import torch and other required modules
import torch
import numpy as np
# Example 1 - working (change this)
li=[
    [
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ],
    [
        [10,11,12],
        [13,14,15],
        [16,17,18]
    ],
    [
        [19,20,21],
        [22,23,24],
        [25,26,27]
    ]
]
torch.tensor(li)
tu=tuple(li)
torch.tensor(tu)
arr=np.array(tu)
torch.tensor(arr)
li1=[
    [
        [1,2],  # Here we have removed 3
        [4,5,6],
        [7,8,9]
    ],
    [
        [10,11,12],
        [13,15], #here we have removed 14
        [16,17,18]
    ],
    [
        [19,20,21],
        [22,23,24],
        [25,26,27]
    ]
]
torch.tensor(li1)

li2=[
    [
        [1,2],  # Here we have removed 3
        [4,5,6],
        [7,8,9]
    ],
    [
        [10,11], # here we have removed 12
        [13,14,15], 
        [16,17,18]
    ],
    [
        [19,21], # Here we have removed 20
        [22,23,24],
        [25,26,27]
    ]
]
torch.tensor(li2)
li3=[
    [
        [1,2],  
        [5,6],
        [8,9]
    ],
    [
        [10,11], 
        [13,15], 
        [16,18]
    ],
    [
        [19,21], 
        [22,23],
        [25,26]
    ]
]
torch.tensor(li3)
# A tensor with values 0 to 9
x=torch.arange(10)

# A tensor with multiples of 3 upto 30(including)
x1=torch.arange(3,31,3) # Since the last value to be considered is 31-1=30

x,x1
y=torch.randn(5)

# Creating a 2D Tensor
y1=torch.randn((3,3))

# Creating a 3D Tensor
y2=torch.randn((2,3,4))

y,y1,y2
# 5 logarithmically spaced values between 2^1 and 2^10
z=torch.logspace(1,10,5,base=2)

# 10 logarithmically spaced values between 10^0 and 10^2
z1=torch.logspace(0,2,steps=10,base=10)

# 2 logarithmically spaced values netween 4^0 and 4^2
z2=torch.logspace(0,2,2,4)

z,z1,z2
# 5 linearly spaced numbers from 1 to 10
a=torch.linspace(1,10,5)

# 5 linearly spaced numbers from 2 to 12
a1=torch.linspace(2,12,6)

a,a1
type(x),type(y),type(z),type(a), type(torch.tensor([]))
v = torch.tensor([0., 0., 0.], requires_grad=True)
v.backward(torch.tensor([0., 2., 3.]))
v.grad
# So here we will never get a Gradient which is zero the same can be extended to always positive gradients and 
# always greater than zero gradients
def f1(grad):
    for i in range(grad.numel()):
        if grad[i]==0:
            grad[i]+=1

v = torch.tensor([0., 0., 0.], requires_grad=True)
h = v.register_hook(f1)
v.backward(torch.tensor([0., 2., 3.]))
v1=v.grad
h.remove()

v2= torch.tensor([0., 0., 0.], requires_grad=True)
v2.backward(torch.tensor([0., 2., 3.]))
v3=v2.grad

v1,v3
# So here Gradient is always greater than zero gradients
def f1(grad):
    for i in range(grad.numel()):
        if grad[i]<0:
            grad[i]*=-1
        elif grad[i]==0:
            grad[i]+=1

v = torch.tensor([0., 1., 0.], requires_grad=True)
h = v.register_hook(f1)
v.backward(torch.tensor([-5., 0., 3.]))
v1=v.grad
h.remove()

v2= torch.tensor([0., 1., 0.], requires_grad=True)
v2.backward(torch.tensor([-5., 0., 3.]))
v3=v2.grad

v1,v3
v=torch.tensor([3.,3.,3.,3.], requires_grad=True)
learning_rate=1
h=v.register_hook(lambda grad:grad*learning_rate)
v.backward(torch.tensor([1.,1.5,3.,6.]))
v.grad
v=torch.tensor([3.,3.,3.,3.], requires_grad=True)
learning_rate=0.1
h=v.register_hook(lambda grad:grad*learning_rate)
v.backward(torch.tensor([1.,1.5,3.,6.]))
v.grad
# Breaking The Function
v=torch.tensor([3.,3.,3.,3.], requires_grad=True)
learning_rate=0.1
h=v.register_hook(lambda x:x[0]*learning_rate)
v.backward(torch.tensor([1.,1.5,3.,6.]))
v.grad
s=torch.tensor(
    [
        [0,0,0],
        [4,0,0],
        [0,1,0]
    ]
)
s.to_sparse()
s.to_sparse(sparse_dim=1)
s.to_sparse(3)
s1=torch.tensor(
    [
        [
            [0,2,0],
            [4,0,0],
            [0,0,0]
        ],
        [
            [0,0,12],
            [0,0,0],
            [16,0,0]
        ],
        [
            [0,0,0],
            [0,0,0],
            [0,0,0]
        ]
    ]
)

s1.to_sparse(3)
t=torch.tensor([1,2,3])
t1=torch.tensor([1,0,0])
t2=torch.tensor([0,1,0])
t3=torch.tensor([0,0,1])
(torch.Tensor.dot(t,t), # 1*1 + 2*2 + 3*3
 torch.Tensor.dot(t,t1), # 1*1 + 0*2 + 0*3
 torch.Tensor.dot(t,t2), # 0*1 + 1*2 + 0*3
 torch.Tensor.dot(t,t3)) # 0*1 + 0*2 + 1*3
ten=torch.tensor([
    [0,1],
    [2,3]
])
torch.Tensor.dot(ten,ten)
ten_sor=torch.tensor([
    [
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ],
    [
        [10,11,12],
        [13,14,15],
        [16,17,18]
    ],
    [
        [19,20,21],
        [22,23,24],
        [25,26,27]
    ]
])
ten_sor.shape
(
    torch.Tensor.flatten(ten_sor,start_dim=1,end_dim=2).shape,# 0th dimension will be untouched
    torch.Tensor.flatten(ten_sor,start_dim=0,end_dim=1).shape # 2nd dimesion will be untouched
)
torch.Tensor.flatten(ten_sor,start_dim=1,end_dim=2) 

#same as torch.flatten(ten_sor,start_dim=-2,end_dim=-1) 

torch.Tensor.flatten(ten_sor,start_dim=0,end_dim=1)

#same as torch.flatten(ten_sor,start_dim=-3,end_dim=-2)
# default #same as torch.flatten(ten_sor,start_dim=-3,end_dim=-1) or 
# same as torch.flatten(ten_sor,start_dim=0 ,end_dim=2)
torch.Tensor.flatten(ten_sor)
torch.Tensor.flatten(ten_sor,start_dim=2,end_dim=0)
torch.Tensor.flatten(ten_sor,start_dim=3)
te=torch.tensor([
    [
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ],
    [
        [10,11,12],
        [13,14,15],
        [16,17,18]
    ],
    [
        [19,20,21],
        [22,23,24],
        [25,26,27]
    ]
])
te.numel()
torch.arange(10).numel()
torch.randn((10,101,8)).numel()
!pip install jovian --upgrade --quiet
import jovian
jovian.commit()
