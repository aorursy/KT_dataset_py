import torch
def foo(x,y):

    return torch.dot(x,y)/(torch.dot(x,x)**0.5*torch.dot(y,y)**0.5)

def grad(x,y):

    x.requires_grad_(True)

    y.requires_grad_(True)

    res=foo(x,y)

    res.backward()

    print(x.grad)

    print(y.grad)

    
x=torch.tensor([1.,2.,3.])

y=torch.tensor([2.,5.,8.])

grad(x,y)

xl=torch.tensor([1.,2.,3.])

yl=torch.tensor([2.,5.,8.])

x1=torch.tensor([0.001,0.,0.])

x2=torch.tensor([0.,0.001,0.])

x3=torch.tensor([0.,0.,0.001])

q1=(foo(xl+x1,yl)-foo(xl-x1,yl))/(2*x1)

q2=(foo(xl+x2,yl)-foo(xl-x2,yl))/(2*x2)

q3=(foo(xl+x3,yl)-foo(xl-x3,yl))/(2*x3)

res=torch.tensor([q1[0],q2[1],q3[2]])

print(res)
def task_2():

    X = torch.tensor([[-1 ,-1],[3, 5.1], [0, 4], [-1, 5], [3, -2], [4, 5]])

    w = torch.tensor([0.1, -0.1])

    b = torch.tensor(0.)

    y = torch.tensor([0.436, 14.0182, 7.278, 6.003, 7.478, 15.833])

    y_hat = torch.matmul(X,w) + b # или torch.sum(X*w,dim=1) + b

    print(y_hat)

    b.requires_grad_(True)

    w.requires_grad_(True)

    lr = 0.001

    for iteration in range(2200):

        with torch.no_grad():

            if w.grad is not None and b.grad is not None: 

                w.grad.zero_()

                b.grad.zero_()

        y_hat = torch.matmul(X,w) + b

        f=torch.dot(y_hat-y,y_hat-y)

        print(w.data, b.data, f.item())

        f.backward()

        with torch.no_grad():

            w -= lr * w.grad

            b -= lr * b.grad

    print(w.data, b.data, f.item())
task_2()
def task_31(p):

    N=4

    D=7

    a=torch.rand(N,D)

    print(a)

    for i in range (N):

        for j in range (D):

            if p>torch.rand(1):

                a[i][j]=torch.zeros(1)

    print(a)
task_31(1)

task_31(0.5)
def task_32(p):

    N=4

    D=7

    a=torch.rand(N,D)

    print(a)

    for j in range (D):

        if p>torch.rand(1):

            for i in range (N):           

                a[i][j]=torch.zeros(1)

    print(a)
task_32(1)

task_32(0.5)
def task_33():

    N=4

    D=7

    K=torch.rand(N,D)

    q=torch.rand(D)

    z=torch.matmul(K,q)

    e=torch.exp(z)

    es=e.sum()

    ai=e/es

    a=torch.matmul(ai,K)

    print(a)
task_33()