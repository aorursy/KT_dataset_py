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
np.random.choice(['cat', 'dog', 'sheep'], size=10)
np.random.choice(['cat', 'dog', 'sheep'], size=10, p=[0.35, 0.5,0.15])
a = np.arange(9)
a
np.where(a%2==0)
a[np.where(a%2==0)]
np.where(a % 2==0, -1, a)
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
# TODO
# TODO
# TODO
# TODO
# TODO
x = np.arange(1,5)
plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)
#TODO
x = np.arange(1,5)
plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)
plt.grid(True)
# TODO
x = np.arange(1,5)
plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)
plt.grid(True)
# TODO
x = np.arange(1,5)
plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)
plt.grid(True)
# TODO
x = np.arange(1,5)
plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)
plt.grid(True)
# TODO
plt.plot([1,3,2,4])
# TODO
x = np.linspace(-np.pi, np.pi, 256)
sin_y = np.sin(x)
cos_y = np.cos(x)
# TODO
x = np.linspace(-np.pi, np.pi, 256)
sin_y = np.sin(x)
cos_y = np.cos(x)
plt.plot(x, sin_y, label="Sin")
plt.plot(x, cos_y, label="cos")
# TODO
%matplotlib inline
x = np.linspace(-np.pi, np.pi, 256)
sin_y = np.sin(x)
cos_y = np.cos(x)
# TODO

x = np.linspace(-np.pi, np.pi, 256)
sin_y = np.sin(x)
cos_y = np.cos(x)
plt.plot(x, sin_y, label="Sin")
plt.plot(x, cos_y, label="cos")
plt.legend()
# TODO
x = np.linspace(-np.pi, np.pi, 256)
sin_y = np.sin(x)
cos_y = np.cos(x)
plt.plot(x, sin_y, label="Sin")
plt.plot(x, cos_y, label="cos")
plt.legend()
# TODO axes

plt.savefig('myplot.png')
# TODO subplot
plt.plot([1,3,2,4])
plt.xlabel("this is x axis")
plt.ylabel("this is y axis")

# TODO subplot
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
# TODO subplot
# TODO plot

#plt.subplot(2,2,2)
#plt.xticks([]), plt.yticks([])
#plt.text(0.5,0.5, 'subplot(2,2,2)',ha='center',va='center',size=20,alpha=.5)

#plt.subplot(2,2,3)
#plt.xticks([]), plt.yticks([])
#plt.text(0.5,0.5, 'subplot(2,2,3)',ha='center',va='center',size=20,alpha=.5)

#plt.subplot(2,2,4)
#plt.xticks([]), plt.yticks([])
#plt.text(0.5,0.5, 'subplot(2,2,4)',ha='center',va='center',size=20,alpha=.5)
# TODO hist plot
# TODO scatter plot
# TODO MM beans
#TODO bar chart
#TODO pie chart
# TODO error bart

def f(x,y):
    return (1-x/2+x**2+y**5)*np.exp(-x**2-y**2)
# TODO
#plt.imshow(f(X,Y))
from mpl_toolkits.mplot3d import Axes3D

#TODO

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





