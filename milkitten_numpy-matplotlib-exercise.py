import numpy as np



lst1 = [0,1,2,3,4]

array1 = np.array(lst1)
type(array1)
array1 + 2
lst2 = [[0,1,2],[3,4,5]]

array2 = np.array(lst2)
array2
array2 = np.array(lst2, dtype='float')

array2
array2 = array2.astype('int')

array2
print('shape is: ', array2.shape)

print('size is: ', array2.size)

print('dtype is: ', array2.dtype)

print('dimension is: ', array2.ndim)
array2[:2, :2]
array2 %2 == 0
array2[array2 % 2==0]
np.nan   #难。。(not a number)
np.inf  #英菲尼迪。。(infinite)
array2 = array2.astype('float')

array2[1,2] = np.nan

array2[0,1] = np.inf

array2
suoyin = np.isnan(array2) | np.isinf(array2)

array2[suoyin] = 666

array2
array2[array2==666] = 0

array2
print('the mean value is: ', array2.mean())

print('the max value is: ', array2.max())

print('the min value is: ', array2.min())
print("max value of each column: ", array2.max(axis=0))

print("min value of each column: ", array2.min(axis=1))
array2
array2_cpy = array2[:, 1:]

array2_cpy
array2_cpy[:1,:1] = 1
array2_cpy
array2
array2[array2==1] = 0

array2_cpy = array2[:,1:].copy()

array2_cpy[:1,:1] = 666

array2_cpy
array2
array2
array2.shape
array2 = array2.reshape(3,2)

array2
array2_ravel_result = array2.ravel()

array2_flatten_result = array2.flatten()
array2_ravel_result
array2_flatten_result
array2_ravel_result[0] = 666

array2
array2_flatten_result = 0

array2
a = np.arange(9)

a
a = a.reshape(3,3)

a
a = np.arange(8,-1,-1)

a

a = a.reshape(3,3)

a
a = np.linspace(0,8,9)

a
a = np.linspace(0,10,5)

a
np.set_printoptions(precision=2)

a = np.logspace(1,10, num=10,base=10)

a
np.log10(a)
a = np.array([1,2,3])

tile_a = np.tile(a, 2)

repeat_a = np.repeat(a, 2)
tile_a
repeat_a
a = np.array(['cat', 'cat', 'dog','sheep','dog'])
np.unique(a)
np.unique(a, return_counts=True)
np.random.seed(19)

np.random.rand(3,3)
np.random.randn(3,3)
np.random.randint(0,10, size=(3,3))
np.random.choice(['cat', 'dog', 'sheep', 'car', 'ship', 'horse'], size = 10)
np.random.choice(['cat', 'dog', 'sheep'], size=10, p=[0.35, 0.5,0.15])
a = np.arange(9)

a = a*2+1

b = a.reshape((3,3))

print(a)

print(b)
print(np.where(a%3==0))

print(b%3==0)

tmp = np.where(a%3==0)

print(type(tmp))

tmp = b%3==0

print(type(tmp))
print(a[np.where(a%3==0)])

print(b[b%3==0])

print(type(a[np.where(a%3==0)]))

print(type(b[b%3==0]))
np.where(a % 3==0, -1, a)
a = np.arange(4).reshape(2,2)

b = np.arange(6,10).reshape(2,2)
a
b
c = np.concatenate([a,b], axis=0)
c.shape
c
d = np.concatenate([a,b], axis=1)
d.shape
d
np.save('temp_result', d)
load_d = np.load('temp_result.npy')
load_d
def myfunc(x):

    if x % 2 == 0:

        return x / 2

    else:

        return x ** 2
myfunc(2),myfunc(5)
a = np.array([1,2,3])
myfunc_v = np.vectorize(myfunc)

myfunc_v(a)
a = np.arange(6).reshape(2,3)

a
def func(x):

    return (max(x) - min(x)) / 2
np.apply_along_axis(func, 0, a)
np.apply_along_axis(func, 1, a)
%matplotlib inline

import matplotlib.pyplot as plt
plt.title('test')

plt.plot([6,1,9,8])

plt.show()
x = range(5)

y = [i**2 for i in x] 

plt.plot(x,y)

plt.show()
x = np.arange(0,4,0.01)

y = [i**2 for i in x] 

plt.plot(x,y)

plt.show()
x = np.linspace(-np.pi, np.pi, 100)

plt.plot(x, np.sin(x))

plt.plot(x, np.cos(x))

plt.show()
plt.plot(x, np.sin(x), x, np.cos(x))

plt.show()
x = np.arange(1,5)

plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)

plt.grid(True)

plt.show()
x = np.arange(1,5)

plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)

plt.grid(True)

plt.axis()
x = np.arange(1,5)

plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)

plt.grid(True)

plt.axis([0,5,0,15])
x = np.arange(1,5)

plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)

plt.grid(True)

plt.axis([1,4,0,12])
x = np.arange(1,5)

plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)

plt.grid(True)

plt.xlim([0,5])

plt.ylim([0,15])
plt.plot([1,3,2,4])

plt.xlabel('x')

plt.ylabel('y')

plt.title('test')
x = np.linspace(-np.pi, np.pi, 256)

sin_y = np.sin(x)

cos_y = np.cos(x)

plt.plot(x, sin_y, label = 'sin')

plt.plot(x, cos_y, label = 'cos')

plt.legend()

plt.show()
x = np.linspace(-np.pi, np.pi, 256)

sin_y = np.sin(x)

cos_y = np.cos(x)

plt.plot(x, sin_y, label="Sin")

plt.plot(x, cos_y, label="cos")

plt.legend(loc = (0.5,0.5))

plt.show()
%matplotlib inline

x = np.linspace(-np.pi, np.pi, 256)

sin_y = np.sin(x)

cos_y = np.cos(x)

plt.plot(x, sin_y, label="Sin", color = 'red', linewidth = '10', linestyle = ':')

plt.plot(x, cos_y, label="cos", color = 'blue', linewidth = '1', linestyle = '-')

plt.legend(loc = (0.1,0.1))

plt.show()

x = np.linspace(-np.pi, np.pi, 256)

sin_y = np.sin(x)

cos_y = np.cos(x)

plt.plot(x, sin_y, label="Sin", color = 'red', linewidth = '10', linestyle = ':')

plt.plot(x, cos_y, label="cos", color = 'blue', linewidth = '1', linestyle = '-')

plt.legend(loc = (0.1, 0.9))

plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])

plt.yticks([-1,0,1])

plt.show()
x = np.linspace(-np.pi, np.pi, 256)

sin_y = np.sin(x)

cos_y = np.cos(x)

plt.plot(x, sin_y, label="Sin")

plt.plot(x, cos_y, label="cos")

plt.legend()

plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],

          [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

plt.yticks([-1,0,1],

          [r'$-1$', r'$0$', r'$+1$'])



plt.savefig('myplot.png')
plt.subplot(2,3,1)

plt.plot([1,3,2,4])

plt.xlabel("this is x axis")

plt.ylabel("this is y axis")



plt.subplot(2,3,6)

x = np.linspace(-np.pi, np.pi, 256)

sin_y = np.sin(x)

cos_y = np.cos(x)

plt.plot(x, sin_y, label="Sin")

plt.plot(x, cos_y, label="cos")

plt.legend()

plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],

          [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

plt.yticks([-1,0,1],

          [r'$-1$', r'$0$', r'$+1$'])

plt.show()
# plt.subplot(2,2,1)

# plt.xticks([0,1,5]), plt.yticks([3,9])

# plt.text(0.5,0.5, 'human\n subplot(2,2,1)',ha='center',va='center',size=20,alpha=.5)

plt.subplot(2,2,1)

plt.xticks([]), plt.yticks([])

plt.text(0.5,0.5, 'human\n subplot(2,2,1)',ha='center',va='center',size=20,alpha=.5)



plt.subplot(2,2,2)

plt.xticks([]), plt.yticks([])

plt.text(0.5,0.5, 'cat\n subplot(2,2,2)',ha='center',va='center',size=20,alpha=.5)



plt.subplot(2,2,3)

plt.xticks([]), plt.yticks([])

plt.text(0.5,0.5, 'dog\n subplot(2,2,3)',ha='center',va='center',size=20,alpha=.5)



plt.subplot(2,2,4)

plt.xticks([]), plt.yticks([])

plt.text(0.5,0.5, 'sheep\n subplot(2,2,4)',ha='center',va='center',size=20,alpha=.5)
y = np.random.randn(1000)

plt.hist(y, 25)

plt.show()
x = np.random.randn(1000)

y = np.random.randn(1000)

plt.subplot(2,1,1)

plt.scatter(x,y)

y = 2*x + 1

plt.subplot(2,1,2)

plt.scatter(x,y)

plt.show()
size = 50 * abs(np.random.randn(1000))

colors = np.random.randn(1000)

plt.scatter(x, y, s=size, c=colors)

plt.show()
plt.bar([1,2,3],[4,2,6])

plt.show()
x = [20,30,50]

labels = ['cat', 'dog', 'sheep']

plt.pie(x, labels=labels)

plt.show()
x = np.arange(0,4,0.2)

y = np.exp(-x)

e = 0.1 * abs(np.random.randn(len(y)))

plt.errorbar(x, y, yerr=e, fmt='.-')

plt.show()
def f(x,y):

    return x**2 + y**2

n = 256

x = np.linspace(-3,3,n)

y = np.linspace(-3,3,n)

X,Y = np.meshgrid(x,y)



print(np.meshgrid(x,y))



plt.contourf(X, Y, f(X,Y), 8, alpha=.75, cmap=plt.cm.hot)
def f(x,y):

    return (1-x/2+x**2+y**5)*np.exp(-x**2-y**2)

plt.imshow(f(X,Y))
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure()

ax = Axes3D(fig)

X = np.arange(-3, 3, 0.25)

Y = np.arange(-3, 3, 0.25)

X, Y = np.meshgrid(X, Y)

Z = X**2 + Y**2



ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')

plt.show()
import numpy as np
def update(frame):

    global P, C, S



    # Every ring is made more transparent

    C[:,3] = np.maximum(0, C[:,3] - 1.0/n)



    # Each ring is made larger

    S += (size_max - size_min) / n



    # Reset ring specific ring (relative to frame number)

    i = frame % 50

    P[i] = np.random.uniform(0,1,2)

    S[i] = size_min

    C[i,3] = 1



    # Update scatter object

    scat.set_edgecolors(C)

    scat.set_sizes(S)

    scat.set_offsets(P)



    # Return the modified object

    return scat,
#%matplotlib notebook

from matplotlib.animation import FuncAnimation





fig = plt.figure(figsize=(6,6), facecolor='white')

ax = fig.add_axes([0,0,1,1], frameon=False, aspect=1)



n = 50

size_min = 50

size_max = 50*50



# Ring position

P = np.random.uniform(0,1,(n,2))



# Ring colors

C = np.ones((n,4)) * (0,0,0,1)

# Alpha color channel goes from 0 (transparent) to 1 (opaque)

C[:,3] = np.linspace(0,1,n)



# Ring sizes

S = np.linspace(size_min, size_max, n)



# Scatter plot

scat = ax.scatter(P[:,0], P[:,1], s=S, lw = 0.5,

                  edgecolors = C, facecolors='None')



# Ensure limits are [0,1] and remove ticks

ax.set_xlim(0,1), ax.set_xticks([])

ax.set_ylim(0,1), ax.set_yticks([])

animation = FuncAnimation(fig, update, interval=10, blit=True, frames=200)

plt.show()

