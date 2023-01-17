import numpy as np  
a = np.array([0,1,2,3,4])
print(f'Numpy Array a \n{a}')

print(f'Type of Numpy Array a {type(a)}')

print(f'Elements Type of Numpy Array a {a.dtype}')

print(f'Size of Numpy Array a {a.size}')

print(f'Dimensions of Numpy Array a {a.ndim}')

print(f'Shape of Numpy Array a {a.shape}')
a[0]
for index,element in enumerate(a):

    print(f'index {index} element {element}')
b=np.array([3.1,11.02,6.2,231.2,5.2])
print(f'Numpy Array b {b}')

print(f'Type of Numpy Array b {type(b)}')

print(f'Elements Type of Numpy Array b {b.dtype}')

print(f'Size of Numpy Array b {b.size}')

print(f'Dimensions of Numpy Array b {b.ndim}')

print(f'Shape of Numpy Array b {b.shape}')
for index,element in enumerate(b):

    print(f'index {index} element {element}')
c=np.array([20,1,2,3,4])

c
c[0]=100

c
c[4]=0

c
d = c[1:4]

d
d.size
c[3:5]=300,400

c
u = np.array([1,0])

u
v = np.array([0,1])

v
z = u + v

z
type(z)
z = u-v

z
y = np.array([1,2])

y
z = 2*y

z
u = np.array([1,2])

u
v = np.array([3,1])

v             
z = u*v

z
u.T
z= np.dot(u,v)

z
z = u@v

z
u = np.array([1,2,3,-1])

u
z = u + 1

z
a = np.array([1,-1,1,-1])

a
mean_a = a.mean()

mean_a
b = np.array([1,-2,3,4,5])

b
max_b = b.max()

max_b
np.pi
x = np.array([0,np.pi/2,np.pi])

x
y = np.sin(x)

y
np.linspace(-2,2,num=5)
np.linspace(-2,2,num=9)
x = np.linspace(0,2*np.pi,100)

x
y = np.sin(x)

y
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(x,y);
A = np.array([[11,12,13],

             [21,22,23],

             [31,32,33]])

print(f'Numpy Array A \n{A}')

print(f'Type of Numpy Array A {type(A)}')

print(f'Elements Type of Numpy Array A {A.dtype}')

print(f'Size of Numpy Array A {A.size}')

print(f'Dimensions of Numpy Array A {A.ndim}')

print(f'Shape of Numpy Array A {A.shape}')
A[0]
A[0][0]
A[1][2]
A[0,0:2]
A[1,:]
A[:,2]
A[0:2,2]
X = np.array([[1,0],

              [0,1]])

X
Y = np.array([[2,1],

              [1,2]])

Y
Z = X + Y

Z
Z = 2*Y

Z
Z = X * Y

Z
Z = X@Y

Z
A = np.array([[0,1,1],

              [1,0,1]])

A
B = np.array([[1,1],

              [1,1],

              [-1,1]])

B
C = A@B

C
# Plotting functions Plotvec1,Plotvec2



def Plotvec1(u, z, v):

    

    ax = plt.axes()

    ax.arrow(0, 0, *u, head_width=0.05, color='r', head_length=0.1)

    plt.text(*(u + 0.1), 'u')

    

    ax.arrow(0, 0, *v, head_width=0.05, color='b', head_length=0.1)

    plt.text(*(v + 0.1), 'v')

    ax.arrow(0, 0, *z, head_width=0.05, head_length=0.1)

    plt.text(*(z + 0.1), 'z')

    plt.ylim(-2, 2)

    plt.xlim(-2, 2)



def Plotvec2(a,b):

    ax = plt.axes()

    ax.arrow(0, 0, *a, head_width=0.05, color ='r', head_length=0.1)

    plt.text(*(a + 0.1), 'a')

    ax.arrow(0, 0, *b, head_width=0.05, color ='b', head_length=0.1)

    plt.text(*(b + 0.1), 'b')

    plt.ylim(-2, 2)

    plt.xlim(-2, 2)
u = np.array([1, 0])

v = np.array([0, 1])

z = u + v
Plotvec1(u, z, v)

print(f"The dot product u@z is {u@z}")
a,b = np.array([-1,1]),np.array([1,1])

Plotvec2(a,b)

print(f"The dot product a@b is {a@b}")
a,b = np.array([1,0]),np.array([0,1])

Plotvec2(a,b)

print(f"The dot product a@b is {a@b}")
# The vectors are perpendicular. 

# As a result, the dot product is zero. 