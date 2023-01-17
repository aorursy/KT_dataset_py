import numpy as np



# regular python list to np.array

lst1 = [0,1,2,3,4] # python list 里面可以放不同类型的数据，np.array 里面不可以

array1 = np.array(lst1)
type(array1)
# python list 是不可以和一个数字相加的，但是np.array可以

array1 + 2
# 2-dim regular python array to np.array

lst2 = [[0,1,2],[3,4,5]]

array2 = np.array(lst2)

array2
# 2-dim regular python array to np.array

lst3 = [[0,3,1],[6,2,3]]

array3 = np.array(lst3)

array3
# define type

array2 = np.array(lst2, dtype='float')

array2
# convert float to int

array2 = array2.astype('int')

array2
print('shape is: ', array2.shape)

# all the items in the array

print('size is: ', array2.size)

print('dtype is: ', array2.dtype)

print('dimension is: ', array2.ndim)
array2
array2[:2, :2]
array2 %2 == 0 # will use this 布尔索引
# pick all the items that %2 ==0 (note: actual items, not the index of the item)

array2[array2 % 2==0] # 布尔索引
array3
array3[array3 % 2==0]
np.nan   #难。。(not a number)
np.inf  #英菲尼迪。。(infinite)
array2 = array2.astype('float') # 必须要从int 转换成 float, 才能做nan, inf赋值，否则会报错

array2[1,2] = np.nan

array2[0,1] = np.inf

array2
suoyin = np.isnan(array2) | np.isinf(array2)

suoyin
array2[suoyin] = 666

array2
array2[array2==666] = 0

array2
# 2 + 3 + 4 = 9, 9 / 6 =1.5

print('the mean value is: ', array2.mean())

print('the max value is: ', array2.max())

print('the min value is: ', array2.min())
# axis = 0: max of each column

print("max value of each column: ", array2.max(axis=0))

# axis = 1: max of each row

print("max value of each row: ", array2.max(axis=1))

array2
array2_cpy = array2[:, 1:]

array2_cpy
# i.e. [0][0] element, array2_cpy里面元素的改变 会 影响到原来数组array2 里的数值

array2_cpy[:1,:1] = 1
array2_cpy
array2
# numpy array deep copy

array2[array2==1] = 0

# with copy() function,是深copy. copy 数组里数据的改变，不会影响原来数组里的值

array2_cpy = array2[:,1:].copy() # numpy array copy 等价于python里deep_copy()

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
array2_ravel_result[0] = 888

array2
array2_flatten_result[1] = 999

array2
array2_flatten_result
a = np.arange(9)

a
a = a.reshape(3,3)

a
# 从8开始，到-1， 不包括-1,所以到0， 步长为-1

a = np.arange(8,-1,-1)

a
a = a.reshape(3,3)

a
# 9 表示9 个元素， 0表示起点，8是终点

a = np.linspace(0,8,9)

a
a = np.linspace(0,10,2)

a
len(a)
a = np.linspace(0,10,5)

a
len(a)
# 开始点为0，结束点为0，元素个数为9

# 0代表10的0次方，0代表10的0次方

a = np.logspace(0,0,9)

a
#开始点和结束点是10的幂，0代表10的0次方，9代表10的9次方, 元素个数为10

a = np.logspace(0,9,10)

a
a = np.logspace(0,9,11)

a
np.set_printoptions(precision=2)

a = np.logspace(1,10, num=10,base=2)

a
# output: i.e. 10^1, 10^2, 10^3, 10^4......10^9, 10^10

# num表示10个点

a = np.logspace(1,10, num=10,base=10)

a
np.log10(a)
a = np.array([1,2,3])

tile_a = np.tile(a, 2)

tile_a_row = np.tile(a, (2, 1))

tile_a_row_col = np.tile(a, (2, 2))

repeat_a = np.repeat(a, 2)
tile_a
tile_a_row
tile_a_row_col
repeat_a
a = np.array(['cat', 'cat', 'dog','sheep','dog'])
np.unique(a)
np.unique(a, return_counts=True)
# numpy.random.seed() 可以使多次生成的随机数相同。

#numpy.random.seed()函数可使得随机数具有预见性，即当参数相同时使得每次生成的随机数相同；

#当参数不同或者无参数时，作用与numpy.random.rand()函数相同，即多次生成随机数且每次生成的随机数都不同



# numpy.random.rand(d0, d1, …, dn)的随机样本位于[0, 1)中。 

np.random.seed(19)

np.random.rand(3,3) # 3*3 matrix, each element follows UNIFORM distrubtion U(0,1)
# numpy.random.randn(d0, d1, …, dn)是从标准正态分布中返回一个或多个样本值。 normal distribution

np.random.randn(3,3) # 3*3 matrix, each element follows NORMAL distrubtion U(0,1)
np.random.randint(0,10, size=(3,3)) # 3*3 matrix, each element is sampled from 0 - 9 uniformly
np.random.choice(['cat', 'dog', 'sheep'], size=10)
np.random.choice(['cat', 'dog', 'sheep'], size=10, p=[0.35, 0.5,0.15])
a = np.arange(9)

a_copy = np.append(a, 12)

a

a_copy
# np.where return index of elements

np.where(a_copy%2==0)
np.where(a%2==0)
a_copy[np.where(a_copy%2==0)]
a
# 符合a % 2==0条件的元素设置为-1， 其他地方设置为原来的值

np.where(a % 2==0, -1, a)
a = np.arange(4).reshape(2,2)

b = np.arange(6,10).reshape(2,2)
a
b
# axis = 0: 在行上面append

# pd.concat([data, pd.get_dummies(data[col], prefix=col)[new_cols]], axis=1) 

# https://www.kaggle.com/jungan/ensemble-adaboost-lambda-unique-dummies-concat

c = np.concatenate([a,b], axis=0)
c.shape
c
# axis = 1: 在列上面append

d = np.concatenate([a,b], axis=1)
d.shape
d
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
# each column: 操作在N 行上

np.apply_along_axis(func, 0, a)
# each row 操作在N 列上

np.apply_along_axis(func, 1, a)
%matplotlib inline

import matplotlib.pyplot as plt
plt.title('your id')

# 1， 3， 2， 4， in y axis

plt.plot([1,3,2,4])
x = range(6)

y = [i**2 for i in x]

plt.plot(x, y)
x = np.arange(0.0, 0.6,0.01)

#x = np.linspace(0.0, 0.6, 61)

y = [i ** 2 for i in x]

plt.plot(x, y)
x = np.linspace(-np.pi, np.pi, 256)

sin_y = np.sin(x)

cos_y = np.cos(x)

plt.plot(x, sin_y)

plt.plot(x, cos_y)
plt.plot(x, sin_y, x, cos_y)
x = np.arange(1,5)

plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)

plt.grid(True)
x = np.arange(1,5)

plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)

plt.grid(True)

plt.axis()
x = np.arange(1,5)

plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)

plt.grid(True)

# x axis: 0 to 5, y axis: -1 to 13

plt.axis([0,5,-1,13])
x = np.arange(1,5)

plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)

plt.grid(True)

plt.axis([2,3,0,10])
x = np.arange(1,5)

plt.plot(x, x * 1.5, x, x * 3.0, x, x / 2.0)

plt.grid(True)

plt.axis([2,3,0,10])

print(plt.xlim(0,5))

print(plt.ylim(-1,13))
plt.plot([1,3,2,4])

plt.xlabel("this is x axis")

plt.ylabel("this is y axis")

plt.title("this is ai camp")
# -3.14 to 3.14, 256个点

x = np.linspace(-np.pi, np.pi, 256)

sin_y = np.sin(x)

cos_y = np.cos(x)

plt.plot(x, sin_y, label="Sin")

plt.plot(x, cos_y, label="cos")

# 显示在左上角的标注

plt.legend()
x = np.linspace(-np.pi, np.pi, 256)

sin_y = np.sin(x)

cos_y = np.cos(x)

plt.plot(x, sin_y, label="Sin")

plt.plot(x, cos_y, label="cos")

# 标注的location

plt.legend(loc=(0.5,0.5))
%matplotlib inline

x = np.linspace(-np.pi, np.pi, 256)

sin_y = np.sin(x)

cos_y = np.cos(x)

plt.plot(x, sin_y, color='red', linewidth=3.0, label="Sin", linestyle=':')

plt.plot(x, cos_y, color='blue', linewidth=3.0, label="cos", linestyle='-')

# 标注显示在左上角的外面一点

plt.legend(loc=(0,1)) # plt.legend(loc=(1,0))就在右下角

x = np.linspace(-np.pi, np.pi, 256)

sin_y = np.sin(x)

cos_y = np.cos(x)

plt.plot(x, sin_y, label="Sin")

plt.plot(x, cos_y, label="cos")

plt.legend()

# xtickets, yticks, 函数可以设置只显示 x,y轴上特点的点的标注

plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])

plt.yticks([-1,0,1])
x = np.linspace(-np.pi, np.pi, 256)

sin_y = np.sin(x)

cos_y = np.cos(x)

plt.plot(x, sin_y, label="Sin")

plt.plot(x, cos_y, label="cos")

plt.legend()

# x,y轴上面的标注用特殊的字符代替

plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],

          [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

plt.yticks([-1,0,1],

          [r'$-1$', r'$0$', r'$+1$'])



plt.savefig('myplot.png') # 可以设置路径
# (2,1,1) 表示 2行，1列维度，中 第一个图

plt.subplot(2,1,1)

# 第一个图中的一些数据点

plt.plot([1,3,2,4])

plt.xlabel("this is x axis")

plt.ylabel("this is y axis")



# (2,1,1) 表示 2行，1列维度，中 第二个图

plt.subplot(2,1,2)

#-pie to +pie, 256个数据点

x = np.linspace(-np.pi, np.pi, 256)

sin_y = np.sin(x)

cos_y = np.cos(x)

plt.plot(x, sin_y, label="Sin")

plt.plot(x, cos_y, label="cos")

# 左上角的标注

plt.legend()

plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],

          [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

plt.yticks([-1,0,1],

          [r'$-1$', r'$0$', r'$+1$'])

plt.show()
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
# randn normal distribution

y = np.random.randn(1000)

plt.hist(y,25)

plt.show()
x = np.random.randn(1000)

y = 3*x + 4#np.random.randn(1000)

plt.scatter(x, y)

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

y = np.exp(-x) # 实际的label值

e = 0.1 * abs(np.random.randn(len(y))) # 误差
plt.errorbar(x, y, yerr=e, fmt='.-')

plt.show()

# y-e, y+e
plt.errorbar(x, y, yerr=e, fmt='.--')

plt.show()
# 和Kaggle kernal: AICamp Ensemble Exercise 1 Complete Version 里的 plot_decision_boundary 函数类似



def f(x,y):

    return (1-x/2+x**2+y**5)*np.exp(-x**2-y**2)



n =  256

x = np.linspace(-3,3,n)

y = np.linspace(-3,3,n)

# np.meshgrid(x,y) return list type

X,Y = np.meshgrid(x,y)

#type(np.meshgrid(x,y))



#plt.axes([0.025,0.025,0.95,0.95])



# camp=plt.cm.Paired cmap=plt.cm.hot

#plt.contourf(X, Y, f(X,Y), 8, alpha=.75, cmap=plt.cm.Paired)

#plt.contourf(X, Y, f(X,Y),cmap=plt.cm.Paired)

plt.contour(X, Y, f(X,Y),cmap=plt.cm.Paired)
plt.contourf(X, Y, f(X,Y), 8, alpha=.75, cmap=plt.cm.hot)
plt.imshow(f(X,Y))
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure()

ax = Axes3D(fig)

X = np.arange(-3, 3, 0.25)

Y = np.arange(-3, 3, 0.25)

X, Y = np.meshgrid(X, Y)

R = np.sqrt(X**2 + Y**2)

Z = np.sin(R)



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
# 用在 BaggedTrees的实现与点集实验 (Ensemble Exercise 1)

def plot_decision_boundary(X, model):

    h = .02 

    # X[:, 0].min(): 第0列 最小的， X[:, 0].max()： 第0列最大的

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    # X[:, 1].min(): 第1列 最小的， X[:, 1].max()： 第1列最大的

    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # x_min, x_max 第0列 最小的和最大的

    # y_min, y_max 第1列 最小的和最大的

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))





    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])



    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

# dummy testing data

x = [[8,9], [3, 2]]

X = np.array(x)

X
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
X[:, 0].min(), X[:, 0].max(), X[:, 1].min(),  X[:, 1].max()
x_min, x_max
y_min, y_max
h = .02

#第0列 min-1, max+1, 步长为0.02

column_0= np.arange(x_min, x_max, h)

column_0
column_0.shape
#第一列 min-1, max+1, 步长为0.02

column_1 = np.arange(y_min, y_max, h)

column_1.shape
xx, yy = np.meshgrid(column_0,column_1)
xx.shape, yy.shape
xx_yy = np.c_[xx.ravel(), yy.ravel()]

xx_yy.shape # i.e. xx.ravel作为第一列， yy.ravel作为第二列
plt.contourf(xx, yy, f(xx,yy), cmap=plt.cm.Paired)
# 因为bb的大小是3， 所以aa 在行上重复三次

aa, bb = np.meshgrid([1, 2],[6,7,8])

aa
# 因为 aa.大小为2， 所以bb里每个元素在列上面重复2词

bb
aa.ravel(), bb.ravel()
# 按行拼接 aa.ravel作为第一列， bb.ravel.作为第二列

aa_bb = np.c_[aa.ravel(), bb.ravel()]

aa_bb.shape
aa_bb
# 按列拼接成矩阵

np.r_[aa.ravel(), bb.ravel()]
len(xx.ravel()) #i.e. 450 * 350 展开
def f(x,y):

    return (1-x/2+x**2+y**5)*np.exp(-x**2-y**2)

