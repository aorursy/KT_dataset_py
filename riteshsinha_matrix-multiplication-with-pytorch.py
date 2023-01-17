%load_ext autoreload

%autoreload 2

%matplotlib inline

import torch
def matmul(a,b):

    arow, acol = a.shape

    brow, bcol = b.shape

    c = torch.zeros(arow, bcol) # creating the output array

    for i in range(arow):

            #print(i)

            for j in range(acol):

                    for k in range(bcol):

                        c[i,k] = a[i,j] * b[j,k]

    return(c)
x = torch.randn(2,1)

y = torch.randn(1,2)
z = matmul(x,y)
print(z)
def matmul(a,b):

    arow, acol = a.shape

    brow, bcol = b.shape

    c = torch.zeros(arow, bcol) # creating the output array

    for i in range(arow):

            #print(i)

            for j in range(acol):

                for k in range(bcol):

                    c[i,k] = (a[i,:] * b[:,k]).sum()

    return(c)
z = matmul(x,y)

print(z)

# The output should confirm to earlier one.
def matmul(a,b):

    arow, acol = a.shape

    brow, bcol = b.shape

    c = torch.zeros(arow, bcol) # creating the output array

    for i in range(arow):

        c[i,:] = (a[i,None] * b).sum(dim = 0)

    return(c)
print(matmul(x,y))
def matmul(a,b): return torch.einsum('ik,kj->ij', a, b)
matmul(x,y)