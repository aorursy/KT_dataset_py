# This gallery contains examples of the many things you can do with Matplotlib.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
x = np.arange(9)
y = np.sin(x)
z = np.cos(x)
# marker数据点样式，linewidth线宽，linestyle线型样式，color颜色
plt.plot(x, y, marker="*", linewidth=3, linestyle="--", color="red")
plt.plot(x, z)

# 设置title名称
plt.title("matplotlib")

# 设置x、y 轴标签
plt.xlabel("height")
plt.ylabel("width")
# 设置图例
plt.legend(["Y", "Z"], loc="upper right")
plt.grid(True) # 显示网格
plt.show()
x = np.random.rand(10)
y = np.random.rand(10)
plt.scatter(x, y)
plt.show()
x = np.arange(10)
y = np.random.randint(0, 30, 10)
plt.bar(x, y)
plt.show()
labels = 'frogs', 'hogs', 'dogs', 'logs'
sizes = 15, 20, 45, 10
colors = 'yellowgreen', 'gold', 'lightskyblue', 'lightcoral'
explode = 0, 0.1, 0, 0.1
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)
plt.axis('equal')
plt.show()
# figsize绘图对象的宽度和高度，单位为英寸，dpi绘图对象的分辨率，即每英寸多少个像素，缺省值为80
plt.figure(figsize=(8,6),dpi=100)

# subplot(numRows, numCols, plotNum)
# 一个Figure对象可以包含多个子图Axes，subplot将整个绘图区域等分为numRows行*numCols列个子区域，按照从左到右，从上到下的顺序对每个子区域进行编号
# subplot在plotNum指定的区域中创建一个子图Axes
A = plt.subplot(2, 2, 1)
plt.plot([0, 1], [0, 1], color="red")

plt.subplot(2, 2, 2)
plt.title("B")
plt.plot([0, 1], [0, 1], color="green")

plt.subplot(2, 1, 2)
plt.title("C")
plt.plot(np.arange(10), np.random.rand(10), color="orange")

# 选择子图A
plt.sca(A)
plt.title("A-A")

plt.show()
mean, sigma = 0, 1
x = mean + sigma * np.random.randn(10000)
plt.hist(x, 50)
# plt.hist(x)
plt.show()
# 数据集即三维点 (x,y) 和对应的高度值，共有256个点。高度值使用一个 height function f(x,y) 生成。 x, y 分别是在区间 [-3,3] 中均匀分布的256个值
# 并用meshgrid在二维平面中将每一个x和每一个y分别对应起来，编织成栅格

def f(x,y):
    # the height function
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)

n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X,Y = np.meshgrid(x, y)

# 接下来进行颜色填充。使用函数plt.contourf把颜色加进去，位置参数分别为：X, Y, f(X,Y)。透明度0.75，并将 f(X,Y) 的值对应到color map的暖色组中寻找对应颜色。
# use plt.contourf to filling contours
# X, Y and value for (X,Y) point
plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)

#  接下来进行等高线绘制。使用plt.contour函数划线。位置参数为：X, Y, f(X,Y)。颜色选黑色，线条宽度选0.5。现在的结果如下图所示，只有颜色和线条，还没有数值Label：
# use plt.contour to add contour lines
C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)

# 其中，8代表等高线的密集程度，这里被分为10个部分。如果是0，则图像被一分为二。
# 最后加入Label，inline控制是否将Label画在线里面，字体大小为10。并将坐标轴隐藏
plt.clabel(C, inline=True, fontsize=10)
plt.xticks(())
plt.yticks(())
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)

plt.show()