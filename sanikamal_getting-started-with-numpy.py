import numpy as np

a = np.arange(20).reshape(4, 5)

a
a.shape
a.ndim
a.dtype.name
a.itemsize
a.size
type(a)
b = np.array([6, 7, 8])

b
type(b)
import numpy as np

a = np.array([2,3,4])

a
a.dtype
b = np.array([1.2, 3.5, 5.1])

b
b.dtype
b = np.array([(1.5,2,3), (4,5,6)])

b
c = np.array([(1.5,2,3), (4,5,6),(4,5,6)])

c
d = np.array( [ [1,2], [3,4] ], dtype=complex )

d
np.zeros( (3,4) )
np.ones( (2,3,4), dtype=np.int16 )                # dtype can also be specified
np.empty( (2,3) )                                 # uninitialized, output may vary
np.arange( 10, 30, 5 )
np.arange( 0, 2, 0.3 )                 # it accepts float arguments
from numpy import pi

np.linspace( 0, 2, 9 )                 # 9 numbers from 0 to 2
x = np.linspace( 0, 2*pi, 100 )        # useful to evaluate function at lots of points

f = np.sin(x)

f
g=np.cos(x)

g
arr = np.arange(6)                         # 1d array

print(arr)
arr2 = np.arange(12).reshape(4,3)           # 2d array

print(arr2)
arr3 = np.arange(24).reshape(2,3,4)         # 3d array

print(arr3)
print(np.arange(10000))
print(np.arange(10000).reshape(100,100))
# np.set_printoptions(threshold=np.nan)
arr1 = np.array( [20,30,40,50,60,70,80] )

arr2 = np.arange( 7 )

arr2
result = arr1-arr2

result
print(arr2**2)

print(10*np.sin(arr1))

print(arr1<56)
X = np.array( [[1,4],

            [5,1]] )

Y = np.array( [[8,0],

            [2,4]] )
# elementwise product

X * Y                      
# matrix product

X @ Y                                        
# another matrix product

X.dot(Y) 
a = np.ones((2,5), dtype=int)

b = np.random.random((2,5))
a *= 3

a
b += a

b
a += b                  # b is not automatically converted to integer type
a = np.ones(3, dtype=np.int32)

b = np.linspace(0,pi,3)
b.dtype.name
c = a+b

c
c.dtype.name
d = np.exp(c*1j)

d
d.dtype.name
a = np.random.random((2,12))

a
a.sum()
a.min()
a.max()
b = np.arange(15).reshape(3,5)

b
# sum of each column

b.sum(axis=0)                                                  
# min of each row

b.min(axis=1)                            
# cumulative sum along each row

b.cumsum(axis=1) 
B = np.arange(3)

B
np.exp(B)
np.sqrt(B)
C = np.array([2., -1., 4.])

np.add(B, C)
a = np.arange(10)**3

a
a[2]
a[2:5]
# equivalent to a[0:6:2] = -1000; from start to position 6, exclusive, set every 2nd element to -1000

a[:6:2] = -1000

a
# reversed a

a[ : :-1]                                 
for i in a:

    print(i**(1/3.0))
def f(x,y):

    return 10*x+y
b = np.fromfunction(f,(5,4),dtype=int)

b
b[2,3]
# each row in the second column of b

b[0:5, 1]                      
# equivalent to the previous example

b[ : ,1]                       
# each column in the second and third row of b

b[1:3, : ]                     
for row in b:

    print(row)
for element in b.flat:

    print(element)
a = np.floor(10*np.random.random((3,5)))

a
a.shape
a.ravel()  # returns the array, flattened
a.reshape(5,3)  # returns the array with a modified shape
a.T  # returns the array, transposed
a.T.shape
a.shape
a
a.resize((5,3))

a
a.reshape(5,-1)
a = np.floor(10*np.random.random((4,4)))

a
b = np.floor(10*np.random.random((4,4)))

b
np.vstack((a,b))
np.hstack((a,b))
from numpy import newaxis

np.column_stack((a,b))     # with 2D arrays
a = np.array([4.,2.])

b = np.array([3.,8.])

np.column_stack((a,b))     # returns a 2D array
np.hstack((a,b))           # the result is different
a[:,newaxis]               # this allows to have a 2D columns vector
np.column_stack((a[:,newaxis],b[:,newaxis]))
np.hstack((a[:,newaxis],b[:,newaxis]))   # the result is the same
np.r_[1:4,0,4]
a = np.floor(10*np.random.random((2,12)))

a
np.hsplit(a,3)   # Split a into 3
np.hsplit(a,(3,4))   # Split a after the third and the fourth column
a = np.arange(12)

b = a            # no new object is created
b is a           # a and b are two names for the same ndarray object
b.shape = 3,4    # changes the shape of a

a.shape
c = a.view()

c is a
c.base is a                        # c is a view of the data owned by a
c.flags.owndata
c.shape = 2,6                      # a's shape doesn't change

a.shape
c[0,4] = 1234                      # a's data changes

a
s = a[ : , 1:3]

s[:] = 10           # s[:] is a view of s. Note the difference between s=10 and s[:]=10

a
d = a.copy()                          # a new array object with new data is created

d is a
d.base is a                           # d doesn't share anything with a
d[0,0] = 9999

a
a = np.arange(12)**2                       # the first 12 square numbers

i = np.array( [ 1,1,3,8,5 ] )              # an array of indices

a[i]                                       # the elements of a at the positions i
j = np.array( [ [ 3, 4], [ 9, 7 ] ] )      # a bidimensional array of indices

a[j]                                       # the same shape as j

palette = np.array( [ [0,0,0],                # black

                      [255,0,0],              # red

                      [0,255,0],              # green

                      [0,0,255],              # blue

                      [255,255,255] ] )       # white

image = np.array( [ [ 0, 1, 2, 0 ],           # each value corresponds to a color in the palette

                    [ 0, 3, 4, 0 ]  ] )
palette[image]                            # the (2,4,3) color image
a = np.arange(12).reshape(3,4)

a
i = np.array( [ [0,1],                        # indices for the first dim of a

                [1,2] ] )

j = np.array( [ [2,1],                        # indices for the second dim

                [3,3] ] )



a[i,j]                                     # i and j must have equal shape
a[i,2]
a[:,j]                                     # i.e., a[ : , j]
time = np.linspace(20, 145, 5)                 # time scale

data = np.sin(np.arange(20)).reshape(5,4)      # 4 time-dependent series

time
data
ind = data.argmax(axis=0)                  # index of the maxima for each series

ind
time_max = time[ind]                       # times corresponding to the maxima



data_max = data[ind, range(data.shape[1])] # => data[ind[0],0], data[ind[1],1]...
time_max
data_max
np.all(data_max == data.max(axis=0))
a = np.arange(5)

a
a[[1,3,4]] = 0

a
a = np.arange(12).reshape(3,4)

b = a > 4

b                                          # b is a boolean with a's shape
a[b]                                       # 1d array with the selected elements
a[b] = 0                                   # All elements of 'a' higher than 4 become 0

a
import numpy as np

import matplotlib.pyplot as plt

def mandelbrot( h,w, maxit=20 ):

    """Returns an image of the Mandelbrot fractal of size (h,w)."""

    y,x = np.ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]

    c = x+y*1j

    z = c

    divtime = maxit + np.zeros(z.shape, dtype=int)



    for i in range(maxit):

        z = z**2 + c

        diverge = z*np.conj(z) > 2**2            # who is diverging

        div_now = diverge & (divtime==maxit)  # who is diverging now

        divtime[div_now] = i                  # note when

        z[diverge] = 2                        # avoid diverging too much



    return divtime

plt.imshow(mandelbrot(400,400))

plt.show()
a = np.arange(12).reshape(3,4)

b1 = np.array([False,True,True])             # first dim selection

b2 = np.array([True,False,True,False])       # second dim selection



a[b1,:]                                   # selecting rows
a[b1]                                     # same thing
a[:,b2]                                   # selecting columns
a[b1,b2]                                  # a weird thing to do
a = np.array([2,3,4,5])

b = np.array([8,5,4])

c = np.array([5,4,6,8,3])

ax,bx,cx = np.ix_(a,b,c)

ax
bx
cx
ax.shape, bx.shape, cx.shape
result = ax+bx*cx

result
result[3,2,4]
a[3]+b[2]*c[4]
def ufunc_reduce(ufct, *vectors):

   vs = np.ix_(*vectors)

   r = ufct.identity

   for v in vs:

       r = ufct(r,v)

   return r
ufunc_reduce(np.add,a,b,c)
import numpy as np

a = np.array([[1.0, 2.0], [3.0, 4.0]])

print(a)
a.transpose()
np.linalg.inv(a)
u = np.eye(4) # unit 4x4 matrix; "eye" represents "I"

u
j = np.array([[0.0, -1.0], [1.0, 0.0]])



j @ j        # matrix product
np.trace(u)  # trace
y = np.array([[5.], [7.]])

np.linalg.solve(a, y)
np.linalg.eig(j)
a = np.arange(30)

a.shape = 2,-1,3  # -1 means "whatever is needed"

a.shape
a
x = np.arange(0,10,2)                     # x=([0,2,4,6,8])

y = np.arange(5)                          # y=([0,1,2,3,4])

m = np.vstack([x,y])                      # m=([[0,2,4,6,8],

                                          #     [0,1,2,3,4]])

xy = np.hstack([x,y])                     # xy =([0,2,4,6,8,0,1,2,3,4])
import numpy as np

import matplotlib.pyplot as plt

# Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2

mu, sigma = 2, 0.5

v = np.random.normal(mu,sigma,10000)

# Plot a normalized histogram with 50 bins

plt.hist(v, bins=50, density=1)       # matplotlib version (plot)

plt.show()
# Compute the histogram with numpy and then plot it

(n, bins) = np.histogram(v, bins=50, density=True)  # NumPy version (no plot)

plt.plot(.5*(bins[1:]+bins[:-1]), n)

plt.show()