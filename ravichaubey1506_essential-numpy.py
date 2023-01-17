import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

#sns.set() #To set style



import warnings

warnings.filterwarnings('ignore')
np.array([10,20,30],dtype=float)
np.zeros(10)
np.zeros((3,3))
np.ones(10)
np.ones((3,3))
np.full(3,3.14)
np.full((3,3),3.14)
np.eye(3)
np.arange(0,10,2)
np.linspace(0,10) #By default 50
np.linspace(0,10,10) #Giving 10 equally spaced value between 0 and 10
np.random.random(5) #Uniformly distributed random number between 0 and 1
np.random.random((3,3))
np.random.randint(0,10) #Random Integer between 0 and 10
np.random.randint(0,10,3)
np.random.randint(0,10,(3,3))
np.random.randn(5) #5 normally distributed numbers
np.random.randn(3,3)
np.empty(3)  #an empty array of default values
np.empty((3,3))
x1=np.random.randint(10,size=6)

x2=np.random.randint(10,size=(3,4))

x3=np.random.randint(10,size=(3,4,5))#3,4*5 matrix 
x1
x2
x3
x3.shape
x3.ndim
x3.size
x3.itemsize #size of each block
x3.nbytes #total memory size
grid=np.arange(0,9).reshape(3,3)
grid
x=np.array([1,2,3])
x.reshape((1,3)) #it converted in to 2D Array
x[np.newaxis,:] #added extra axis rowise
x[:,np.newaxis] #Added extra axis columnwise
x=x

y=np.array([3,2,1])
np.concatenate([x,y]) #Combining 2 array together
grid=np.array([[1,2,3],

              [4,5,6]])

grid
np.concatenate([grid,grid])
np.concatenate([grid,grid],axis=1)
np.vstack([x,grid])
y=np.array([[99],

           [99]])
np.hstack([grid,y])
arr=np.arange(10)

arr
np.split(arr,[3,5])  #Spliting intial array on index 3 and 5
grid=np.arange(16).reshape((4,4))

grid
upper,lower=np.vsplit(grid,[2])

print(upper)

print("\n")

print(lower)
left,right=np.hsplit(grid,[2])

print(left)

print("\n")

print(right)
np.add(x,2)  #it is similar to x+2, Universal Functions
x=np.array([5-2j,4-3j])  #abs will return magnitude of complex number

np.abs(x)
x=np.arange(5)

y=np.empty(5)



np.multiply(x,10,out=y) #multiplying x with 10 and then putting the answer in y to save memory



y
y=np.zeros(10)



np.power(2,x,out=y[::2])



y
x=np.arange(1,6)

x
np.add.reduce(x)  #reducing array up to end of element and adding simuntanously
np.add.accumulate(x) #we can see how it is accumulating to result
np.multiply.reduce(x)
np.multiply.accumulate(x)
np.cumsum(x)  #Cummulative sum
np.cumprod(x)
x=np.arange(1,6)

x
np.multiply.outer(x,x) #See the pattern to find out what it is doing
m=np.random.random((3,4))

m
m.sum()   #returning whole sum
m.sum(axis=0)  #sum of all colums
m.sum(axis=1) #along row
a=np.array([1,2,3])

b=np.array([4,5,6])



a+b
a+5 #adding scaller to another size array
arr=np.ones((3,3))

arr
arr+a #adding 1,2,3 in each row, broadcasting smaller array over higher dimension
a=np.arange(3)

b=np.arange(3)[:,np.newaxis]
print(a)

print("\n")

print(b)
a+b #adding 0 to first column of b then 1 to first column and so on
import matplotlib.pyplot as plt

%matplotlib inline



x=np.linspace(0,10,50)

y=np.linspace(10,20,50)[:,np.newaxis]

z=x**2+y**2



plt.imshow(z,cmap='viridis')
x=np.array([1,2,3,4,5])

x
2*x==x**2
x=np.random.randint(10,size=(3,4))

x
x<6
np.count_nonzero(x<6) #9 Values less than 6
np.sum(x<6,axis=0) #counting true x<6 columnwise
np.sum(x<6,axis=1) #Counting <6 rowise
np.any(x<6) #Are there any value less than 6?
np.all(x<6) #Are all the value less than 6?
np.any(x<6,axis=0) #Are there any value less than 6 in each column?
np.any(x<6,axis=1) #Are there any value less than 6 in each row?
x[x<5]  #selecting only those value with less than 5
bool(42)
bool(0)
bool(42) and bool(0)
bool(42) or bool(0)
bin(42)
bin(59)
bin(42 & 59)
bin(42 | 59)
A=np.array([0,1,0,1,0,1])

B=np.array([1,0,1,0,1,0])

print(A)

print("\n")

print(B)
A & B #this is okay because we are applying on bits, but using A and B will raise an error
#A and B #This error is because we are applyinh and operator on bits which is impossible.
x=np.random.randint(100,size=10)

x
x[[1,2,6]] #Printing value at index 1,2 and 6
index=np.array([[3,7],

                [4,5]])

index
x[index] #it is filling value at these position from value at those position in that array
x=np.arange(0,12).reshape((3,4))

x
x[2,[2,0,1]] #selecting from 2nd row, 2th,0th & 1th column value
x=np.random.randn(100,2)

x[:5,:]
indices=np.random.choice(x.shape[0],20,replace=False) #Selecting 20 random points between 0 to 100

indices
sub_array=x[indices,:] #We can see we only have those indices which we have given

sub_array
x=np.arange(10)

i=np.array([2,1,8,4])

x[i]=99  #Changing value at 2th,1th,8th and 4th position to 99

print(x)
i=[2,3,3,4,4,4]

x=np.zeros(10)

np.add.at(x,i,1)  #Add 1 at ith position

print(x)
i=[2,3,3,4,4,4]

x=np.zeros(10)

x[i]+=1  #Add 1 at ith position, But here addition is not happening multiple times

print(x)
np.random.seed(42)

x=np.random.randn(100)

x[0:5]



#Compute Histogram by Hand

bins=np.linspace(-5,5,20)

counts=np.zeros_like(bins) #Returns array like bins with zeros



#finding an appropriate bin for each x

i=np.searchsorted(bins,x)



np.add.at(counts,i,1)



plt.plot(bins,counts,ls='steps')
sns.distplot(x,bins=20,kde=False) #You can see same work with just one line of code
def selection_sort(x):

    for i in range(len(x)):

        swap=i+np.argmin(x[i:])

        (x[i],x[swap])=(x[swap],x[i])

    return x
x=np.array([40,30,20,10])

selection_sort(x)

print(x)
x=np.array([45,52,56,2,3,4])

np.sort(x) #Note this is just printing, not changing intial array
x.sort() #Now this will change initial array

print(x)
x=np.random.randint(10,size=10)

x
np.argsort(x) #This is returning indices of sorted array in increasing order
print(x.argsort(),"\n")

print("Please do not confuse that initial x will change in this also")

print(x)
x=np.random.randint(10,size=(6,4))

x
np.sort(x,axis=0) #Along column
np.sort(x,axis=1) #Along Row
x.sort(axis=0) #This will change initial array also

x
x=np.random.randint(10,size=10)

x
np.partition(x,3) #First 3 elements are smallest and after that in arbitrary order
x=np.random.randint(10,size=(4,6))

x
np.partition(x,3,axis=1) #ALong row
np.partition(x,3,axis=0) #Along Column
np.argpartition(x,3,axis=1) #Similar to argsort but returning in partition