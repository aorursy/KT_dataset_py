%load_ext autoreload
%autoreload 2

%matplotlib inline
#export
# from exp.nb_00 import *
import operator

def test(a,b,cmp,cname=None):
    if cname is None: cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def test_eq(a,b): test(a,b,operator.eq,'==')
#export
from pathlib import Path
from IPython.core.debugger import set_trace
from fastai import datasets
import pickle, gzip, math, torch, matplotlib as mpl
import matplotlib.pyplot as plt
from torch import tensor

MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'
path = datasets.download_data(MNIST_URL, ext='.gz'); path
with gzip.open(path, 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
x_train,y_train,x_valid,y_valid = map(tensor, (x_train,y_train,x_valid,y_valid))
n,c = x_train.shape
x_train, x_train.shape, y_train, y_train.shape, y_train.min(), y_train.max()
assert n==y_train.shape[0]==50000 #chek that n have the sam shape as y_train[0]=50000
test_eq(c,28*28)                  #test_eq is a self made fuction defined in the beginning and here we check if the columns are 28*28
test_eq(y_train.min(),0)          #check that y_train.mib are 0 
test_eq(y_train.max(),9)          #check that y_train.max are 9
# print a image
mpl.rcParams['image.cmap'] = 'gray' 
#first image
img = x_train[0]

#image size
img.view(28,28).type()
#show image
plt.imshow(img.view((28,28)));
weights = torch.randn(784,10) #784 by 10 matrix because we got 748 comming in and 10 going out
bias = torch.zeros(10) #for bias we are just staring with 10 zeros 
##almost everything we do in deep learning are matrix multipication or somethig close to it 
# This will only work if the number of rows on one matrix is the same as the number of columns of the secound matrix 

def matmul(a,b):
    ar,ac = a.shape # n_rows * n_cols: ar=row, ac=columns
    br,bc = b.shape #br=row, bc=columns
    # number of columns in 1. matrix == number of rows in 2. matrix
    assert ac==br
    c = torch.zeros(ar, bc) #then lets create a new matrix of size ar and bc so it has enough columns and rows with zeros in
    for i in range(ar): #for each row in 'a' matrix
        for j in range(bc): #for each column in 'b' matrix
            for k in range(ac): # for each column in 'a' matrix
                c[i,j] += a[i,k] * b[k,j] #This is the part that does the actual calulation where fx where the first row in 'a'
                # are [1,2,1] an the firs column in 'b' are [2,6,1] now the column in ac wich just is for the first row. 
                # the calulation will then look like this ?c[1,2]+=a[1,1]*b[1,2]? if (i´m) confused look at the pictures above 
    return c
x_valid
m1 = x_valid[:5] #take the 5 first rows in validaionse 
m2 = weights  #take the weight matrix 
m1.shape,m2.shape #show shape
%time t1=matmul(m1, m2) #marix mutiplication
t1.shape #here we see that it go the input shape af m1 and the output shape of 10 by m2 in other words 5 rows by 10 columns output

len(x_train)
a = tensor([10., 6, -4])  #we create to tensors with 3 dimensions 
b = tensor([2., 8, 7])
a,b
a + b #then we add hem together an we see that each row are being added together
(a < b).float().mean()
m = tensor([[1., 2, 3], [4,5,6], [7,8,9]]); m
(m*m).sum().sqrt()  #this is the code of the above formular could also be writen as m.pow(2).sum().sqrt()
def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            # Any trailing ",:" can be removed
            c[i,j] = (a[i,:] * b[:,j]).sum() #a[i,:] are all the rows and b[:,j] are all the columns 
    return c
%timeit -n 10 _=matmul(m1, m2) 
890.1/5 #it is faster since the above formular uses C
#export test if the floats are neer since we cant just compare floats with eachother 
def near(a,b): return torch.allclose(a, b, rtol=1e-3, atol=1e-5)
def test_near(a,b): test(a,b,near)
test_near(t1,matmul(m1, m2))
a
a > 0
a + 1
m
2*m
c = tensor([10.,20,30]); c
m
m.shape,c.shape
m + c
c + m
t = c.expand_as(m)
t
m + t
t.storage()
t.stride(), t.shape
c.unsqueeze(0)
c.unsqueeze(1)
m
c.shape, c.unsqueeze(0).shape,c.unsqueeze(1).shape  #you can change rows to columns and change i dimentions hereby
c.shape, c[None].shape,c[:,None].shape  #None means squeeze a new axis in here and note tha unsqueeze and None does he same
c[None].shape,c[...,None].shape
c[:,None].expand_as(m)
m + c[:,None]
c[:,None]
def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
#       c[i,j] = (a[i,:]          * b[:,j]).sum() # previous
        c[i]   = (a[i  ].unsqueeze(-1) * b).sum(dim=0)
        # c[i] set the entyre row same as c[i,:] 
        #.unsqueeze(-1) changes it to a rank two tensor -1 because it is the last dimension same as a[i, None] so it 
        # is of shape ar , 1 since a[i  ].unsqueeze(-1) is the rows. And 'b' is the entyraty of our tensor so it is also rank 2
        #this return a rank 2 tensor but will sum it up over the rows so we use .sum(dim=0) (dim=0 wha axis you want to sum over)
        
    return c
%timeit -n 10 _=matmul(m1, m2)
885000/277
test_near(t1, matmul(m1, m2))
c[None,:]
c[None,:].shape
c[:,None]
c[:,None].shape
c[None,:] * c[:,None]
c[None] > c[:,None]
# c[i,j] += a[i,k] * b[k,j]
# c[i,j] = (a[i,:] * b[:,j]).sum()
def matmul(a,b): return torch.einsum('ik,kj->ij', a, b)
%timeit -n 10 _=matmul(m1, m2)
885000/55
test_near(t1, matmul(m1, m2))
%timeit -n 10 t2 = m1.matmul(m2)
# time comparison vs pure python:
885000/18
t2=m1@m2 # '@' means matrix multiplication 
t2.shape 