# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
np.random.seed(12345)
import matplotlib.pyplot as plt

plt.rc('figure',figsize=(10,6))
np.set_printoptions(precision=4, suppress=True)
my_arr=np.arange(1000000)

my_list=(list(range(1000000)))
%time for _ in range(10): my_arr2=my_arr*2

%time for _ in range(10): my_list2=[x*2 for x in my_list]
data=np.random.randn(2,3)

data
data*10

data+data
data.shape

data.dtype
#creating ndarrays

data1=[6,7.5,8,0,1]

arr1=np.array(data1)

arr1
data2=[[1,2,3,4],[5,6,7,8]]

arr2=np.array(data2)

arr2
arr2.ndim
arr2.shape
arr1.dtype

arr2.dtype
np.zeros(10)

np.zeros((3,6))

np.empty((2,3,2))
np.arange(15)
#datatypes for ndrrays

arr1=np.array([1,2,3],dtype=np.float64)

arr2=np.array([1,2,3],dtype=np.int32)

arr1.dtype

arr2.dtype
arr=np.array([1,2,3,4,5])

arr.dtype

float_arr=arr.astype(np.float64)

float_arr.dtype
arr=np.array([3.7,-1.2,-2.6,0.5,12.9,10.1])

arr

arr.astype(np.int32)
numeric_strings=np.array(['1.23','-9.6','42'],dtype=np.string_)

numeric_strings.astype(float)
int_array=np.arange(10)

calibers=np.array([.22,.270,.357,.360,.44,.50],dtype=np.float64)

int_array.astype(calibers.dtype)
empty_uint=np.empty(8,dtype='u4')

empty_uint
#arithmetic withnumpy arrays

arr=np.array([[1.,2.,3.],[4.,5.,6.]])

arr

arr+arr

arr*arr

arr-arr

arr
1/arr

arr**0.5


arr2=np.array([[0.,4.,1.],[7.,2.,12.]])

arr2

arr2>arr
#basic indexing and slicing

arr=np.arange(10)

arr

arr[5]

arr[5:8]
arr[5:8]=12

arr
arr_slice=arr[5:8]

arr_slice
arr_slice[1]=12345

arr
arr_slice[:]=64

arr
arr2d=np.array([[1,2,3],[4,5,6],[7,8,9]])

arr2d[2]
arr2d[0][2]

arr2d[0,2]
arr3d=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])

arr3d
arr3d[0]
old_values=arr3d[0].copy()
arr3d[0]=42

arr3d

arr3d[0]=old_values

arr3d
arr3d[1,0]
x=arr3d[1]

x

x[0]
#indexing ith slices

arr

arr[1:6]
arr2d

arr2d[:2]
arr2d[:2,1:]
arr2d[1,:2]
arr2d[:2,2]
arr2d[:,:1]
arr2d[:2,:1]=0

arr2d
#Boolean indexing

names=np.array(['Bob','joe','will','joe','will','bob','joe'])
data=np.random.randn(7,4)
names
data
names=='Bob'
data[names=='Bob']
data[names=='Bob',2:]
data[names=='bob',3]
names!='Bob'
data[~(names=='Bob')]
cond=names=='Bob'

data[~cond]
mask=(names=='will')|(names=='joe')
mask
data[mask]
data[data<0]=0
data
data[names!='joe']=7

data
#fancy indexing

arr=np.empty((8,4))
for i in range(8):

    arr[i]=i

arr
arr[[4,3,0,6]]
arr[[-3,-5,-7]]
arr=np.arange(32).reshape((8,4))

arr
arr[[1,5,7,2]][:,[0,3,1,2]]
arr[[1,5,7,2]][0,3,1,2]
#transposing arrats and swapping axes

arr=np.arange(15).reshape((3,5))

arr
arr

arr.T
arr=np.random.randn(6,3)

arr

np.dot(arr.T,arr)
arr=np.arange(16).reshape(2,2,4)

arr
arr.transpose(1,0,2)
arr

arr.swapaxes(1,2)
#universal functions fast element wie array functions

arr=np.arange(10)

arr

np.sqrt(arr)

np.exp(arr)
x=np.random.randn(8)

y=np.random.randn(8)

x

y

np.maximum(x,y)
arr=np.random.randn(7)*5

arr

remainder,whole_part=(np.modf(arr))
remainder

whole_part
arr

np.sqrt(arr)

np.sqrt(arr,arr)
arr
#array oriented programming with arrays
points=np.arange(-5,5,0.01)
xs,ys=np.meshgrid(points,points)

ys
z=np.sqrt(xs**2+ys**2)

z
import matplotlib.pyplot as plt

plt.imshow(z,cmap=plt.cm.gray);plt.colorbar()

plt.title("image plot of $\sqrt{x^2+y^2}$ for a grid of values")
plt.draw()
plt.close('all')
#expressing conditional logic as array operations
xarr=np.array([1.1,1.2,1.3,1.4,1.5])

yarr=np.array([2.1,2.2,2.3,2.4,2.5])
cond=np.array([True,False,True,True,False])
result =[(x if x else y) for x,y,c in zip(xarr,yarr,cond)]

result
result=np.where(cond,xarr,yarr)

result
arr=np.random.randn(4,4)

arr
arr>0

np.where(arr>0,2,-2)
np.where(arr>0,2,arr)
#mathematical and statistical methods
arr=np.random.randn(5,4)

arr
arr.mean()
np.mean(arr)
arr.sum()
arr.mean(axis=1)
arr.sum(axis=0)
arr=np.array([0,1,2,3,4,5,6,7,])

arr.cumsum()
arr=np.array([[0,1,2],[3,4,5],[6,7,8]])

arr
arr.cumsum(axis=0)
arr.cumsum(axis=1)
arr.cumprod(axis=1)
#methods for booleans arrays
arr=np.random.randn(100)

(arr>0).sum()
bools=np.array([False,False,True,False])

bools.any()
bools.all()
#sorting

arr=np.random.randn(6)

arr
arr.sort()

arr
arr=np.random.randn(5,3)

arr
arr.sort(1)
arr
large_arr=np.random.randn(1000)

large_arr.sort()

large_arr[int(0.05*len(large_arr))]
#unique and other set logic
names=np.array(['bob','joe','will','bob','will','joe','joe'])

np.unique(names)

ints=np.array([3,3,3,2,2,1,1,4,4])

np.unique(ints)
sorted(set(names))
values=np.array([6,0,0,3,2,5,6])

np.in1d(values,[2,3,6])
#file input and OUput with arrays
arr=np.arange(10)

np.save('some_array',arr)
np.load('some_array.npy')
np.savez('array_archive.npz',a=arr,b=arr)
arch=np.load('array_archive.npz')

arch['b']
np.savez_compressed('arrays_compressed.npz',a=arr,b=arr)
!rm some_array.npy

!rm array_archive.npz

!rm arrays_compressed.npz
#linear algebra
x=np.array([[1.,2.,3.],[4.,5.,6.]])
y=np.array([[6.,23.],[-1,7],[8,9]])
x
y
x.dot(y)
np.dot(x,y)
np.dot(x,np.ones(3))
x @ np.ones(3)
from numpy.linalg import inv,qr

X=np.random.randn(5,5)

mat=X.T.dot(X)

inv(mat)

mat.dot(inv(mat))
q,r=qr(mat)
r
#pseudorandom number generation

samples=np.random.normal(size=(4,4))

samples
from random import normalvariate

N=100000
%timeit samples=[normalvariate(0,1) for _ in range(N)]

%timeit np.random.normal(size=N)
rng=np.random.RandomState(1234)

rng.randn(10)
#example random walks
import random

position=0

walk=[position]

steps=1000

for i in range(steps):

    step=1 if random.randint(0,1) else -1

    position+=step

    walk.append(position)
plt.figure()
plt.plot(walk[:100])
np.random.seed(1345)
nsteps=1000

draws=np.random.randint(0,2,size=nsteps)

steps=np.where(draws>0,1,-1)

walk=steps.cumsum()
walk.min()
walk.max()
(np.abs(walk)>=10).argmax()
#simulating many random walks at once
nwalks=5000

nsteps=1000
draws=np.random.randint(0,2,size=(nwalks,nsteps))
steps=np.where(draws>0,1,-1)

walks=steps.cumsum(1)

walks
walks.max()
walks.min()
hits30=(np.abs(walks)>=30).any(1)

hits30

hits30.sum()
crossing_times=(np.abs(walks[hits30])>=30).argmax(1)
crossing_times.mean()
steps=np.random.normal(loc=0,scale=0.25,size=(nwalks,nsteps))
steps