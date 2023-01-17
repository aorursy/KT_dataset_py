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
print("max value of each column: ", array2.max(axis=0)) #注意axis=0是按列排

print("min value of each column: ", array2.min(axis=1)) #axis=1是按行排
array2
array2_cpy = array2[:, 1:]

array2_cpy
array2_cpy[:1,:1] = 1
array2_cpy # 改copy的值不会影响到原来的
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
array2_ravel_result[0] = 666 #改变ravel会影响原来的array

array2
array2_flatten_result = 0 #改变faltten不会影响原来的array

array2
a = np.arange(9)

a
a = a.reshape(3,3)

a
a = np.arange(8,-1,-1)

a
a = a.reshape(3,3)

a
a = np.linspace(0,8,9) # linspace = linear space(start,end,split)

a
a = np.linspace(0,10,5)

a
np.set_printoptions(precision=2)

a = np.logspace(1,10, num=10,base=10) # logspace(start, end, split, base) 从start到end的n个数取以base为底的对数

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

np.random.rand(3,3) #rand返回（0，1）之间的随机数
np.random.randn(3,3) #randn返回具有正态分布的一组样本
np.random.randint(0,10, size=(3,3)) #randint返回给定范围里的整数
np.random.choice(['cat', 'dog', 'sheep'], size=10)
np.random.choice(['cat', 'dog', 'sheep'], size=10, p=[0.35, 0.5,0.15])
a = np.arange(9)

a
np.where(a%2==0) #返回的是下标/位置
a[np.where(a%2==0)]
np.where(a % 2==0, -1, a) # 符合条件的执行option1，不符合的执行option2
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
myfunc_v = np.vectorize(myfunc) #vectorize把函数向量化，可以apply到向量的每一个数里

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

plt.title('random int')

plt.plot(np.random.randint(0,10,5))
# TODO

x = range(1,6)

plt.plot(x, [i**2 for i in x])  # 为什么y=x**2不可以
# TODO

y = [i**2 for i in x]

plt.plot(x,y)
# TODO

x = np.linspace(-np.pi,np.pi,500)

sinx = np.sin(x)

cosx = np.cos(x)

plt.plot(x,sinx)

plt.plot(x,cosx)
# TODO

plt.plot(x,sinx,x,cosx)
x = np.arange(1,5)

plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)

#TODO

plt.grid()
x = np.arange(1,5)

plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)

plt.grid(True)

# TODO

plt.axis()
x = np.arange(1,5)

plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)

plt.grid(True)

# TODO

plt.axis([0,5,-1,13])
x = np.arange(1,5)

plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)

plt.grid(True)

# TODO

plt.axis([1,4,2,10])
x = np.arange(1,5)

plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)

plt.grid(True)

# TODO

plt.xlim(0,5)

plt.ylim(-1,13)
plt.plot([1,3,2,4])

# TODO

plt.xlabel('x_label')

plt.ylabel('y_label')

plt.title('my plot')
x = np.linspace(-np.pi, np.pi, 256)

sin_y = np.sin(x)

cos_y = np.cos(x)

# TODO

plt.plot(x,sin_y,label='sin')

plt.plot(x,cos_y,label='cos')

plt.legend()
x = np.linspace(-np.pi, np.pi, 256)

sin_y = np.sin(x)

cos_y = np.cos(x)

plt.plot(x, sin_y, label="Sin")

plt.plot(x, cos_y, label="cos")

# TODO

plt.legend(loc='lower right')
%matplotlib inline

x = np.linspace(-np.pi, np.pi, 256)

sin_y = np.sin(x)

cos_y = np.cos(x)

# TODO

plt.plot(x,sin_y,linewidth = 0.5, color = 'red', linestyle='dashdot', label='sin')

plt.plot(x,cos_y,linewidth = 0.8, color = 'blue', label='cos')

plt.legend()
x = np.linspace(-np.pi, np.pi, 256)

sin_y = np.sin(x)

cos_y = np.cos(x)

plt.plot(x, sin_y, label="Sin")

plt.plot(x, cos_y, label="cos")

plt.legend()

# TODO

plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])

plt.yticks([-1,0,1])
x = np.linspace(-np.pi, np.pi, 256)

sin_y = np.sin(x)

cos_y = np.cos(x)

plt.plot(x, sin_y, label="Sin")

plt.plot(x, cos_y, label="cos")

plt.legend()

# TODO axes

plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],

          [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

plt.yticks([-1,0,1],

          [r'$-1$', r'$0$', r'$+1$'])



plt.savefig('myplot.png')
plt.subplot(2,1,1)

# TODO subplot

plt.plot([1,3,2,4])

plt.xlabel("this is x axis")

plt.ylabel("this is y axis")



plt.subplot(2,1,2)

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

plt.subplot(2,2,1)

plt.xticks([]), plt.yticks([])

plt.text(0.5,0.5, 'subplot(2,2,1)',ha='center',va='center',size=20,alpha=.5)



plt.subplot(2,2,2)

plt.xticks([]), plt.yticks([])

plt.text(0.5,0.5, 'subplot(2,2,2)',ha='center',va='center',size=20,alpha=.5)



plt.subplot(2,2,3)

plt.xticks([]), plt.yticks([])

plt.text(0.5,0.5, 'subplot(2,2,3)',ha='center',va='center',size=20,alpha=.5)



plt.subplot(2,2,4)

plt.xticks([]), plt.yticks([])

plt.text(0.5,0.5, 'subplot(2,2,4)',ha='center',va='center',size=20,alpha=.5)
# TODO hist plot

np.random.seed(0)

y = np.random.randn(100)

plt.hist(y,10)

plt.show()
# TODO scatter plot

np.random.seed(0)

x = np.random.rand(100)

y = 4*x + 12

plt.scatter(x,y)
# TODO MM beans

size = 50 * abs(np.random.randn(1000))

colors = np.arctan2(y,x) 

plt.scatter(x, y, s= size, c=colors)

plt.show()
#TODO bar chart

#matplotlib.pyplot.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)



degree = ['high school', 'bachelor', 'graduate']

counts = [8,15,4]



plt.bar(degree, counts)

plt.show()

#TODO pie chart



plt.pie(counts, labels = degree, autopct='%.2f' )

plt.show()
# TODO error bart

np.random.seed(0)

x = np.random.randint(0,200, 15)

y = 3*x

error = 50 * abs(np.random.randn(15))



plt.errorbar(x,y, yerr = error, fmt='.-', ecolor = 'orange')

plt.show()
# meshgrid函数将两个输入的数组x和y进行扩展，前一个的扩展与后一个有关，后一个的扩展与前一个有关，前一个是竖向扩展，后一个是横向扩展。

# y的大小为5，所以x竖向扩展为原来的5倍，而x的大小为7，所以y横向扩展为原来的7倍。

# 返回的两个matrix的大小是一样的，大小为(y行，x列)



# meshgrid参考内容：https://blog.csdn.net/sinat_29957455/article/details/78825945



x = np.arange(-3, 4, 1) #7个数

y = np.arange(-2, 3, 1) #5个数

X, Y = np.meshgrid(x, y)

print(X,'\n\n',Y)
# 等高线 contour([X, Y,] Z, [levels], **kwargs) levels是等高线数量



def f(x,y):

    return (1-x/2+x**2+y**5)*np.exp(-x**2-y**2)



n = 256

x = np.linspace(-3,3,n)

y = np.linspace(-3,3,n)

X,Y = np.meshgrid(x,y)



plt.contourf(X, Y, f(X,Y), cmap = plt.get_cmap('PuBuGn'))

#colormap列表:https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html



plt.show()
plt.imshow(f(X,Y))
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure() #定义图像窗口

ax = Axes3D(fig) #增加3D坐标

X = np.arange(-4, 4, 0.25)

Y = np.arange(-4, 4, 0.25)

X, Y = np.meshgrid(X, Y)

R = np.sqrt(X**2 + Y**2)

Z = np.sin(R)



# stride是网格跨度，r和c分别表示row和column

# The surface is made opaque by using antialiased=False.

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow')) 



fig.colorbar(surf, shrink=0.6, aspect=5)



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