# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
x = np.arange(0, 2 * np.pi, 0.1)

y = np.sin(x)

plt.plot(x, y, 'r')

plt.show()
x = np.arange(0, 10, 1) # 返回给定间隔区间均匀间隔的值

y = np.random.rand(10, 2) # 返回 [0,1] 均匀分布的随机值



fig, ax = plt.subplots()

ax.plot(x, y[:, 0], label='A')

ax.plot(x, y[:, 1], label='B')



# 标题

plt.title('title')



# 坐标轴标签

plt.xlabel('xlabel')

plt.ylabel('ylabel')



# 坐标边界

plt.xlim([0, 15])

plt.ylim([0, 1.2])



# 坐标刻度

plt.xticks(x)

plt.yticks([0, 0.2, 0.5, 0.7, 0.8, 0.9, 1.2])



# 坐标刻度标签

ax.set_xticklabels(x for x in 'abcdefgh123')

# ax.set_yticklabels([0, 0.2, 0.5, 0.7, 0.8, 0.9, 1.2])

plt.show()
# 直接使用 pd 绘制

# .cumsum() 返回该列的累计和，是一列！

s = pd.Series(np.random.randn(10).cumsum()) 

s.plot(linestyle='--', marker='*', color='g', alpha=0.2, grid=True)

plt.show()
# 直接用style 设置线型，符号

# 颜色板

df = pd.DataFrame(np.random.randn(100,4), columns=list('ABCD')).cumsum()

df.plot(style='--.', alpha=0.9, colormap='summer_r')

plt.show()
x = np.arange(0, 10, 1)

y = np.random.randn(10, 2)

plt.plot(x, y[:, 0], '--o', label='a')

plt.plot(x, y[:, 1], '--o', label='b')

plt.text(4, 0.4, 'text', fontsize=12, color='r') # 4,0.4 这个位置图注

plt.show()
fig1 = plt.figure(num=1, figsize=(8, 6)) # num = id？相同会绘制在同一张图上

plt.plot(np.random.rand(50).cumsum(), 'k--')

fig2 = plt.figure(num=2, figsize=(8, 6))

plt.plot(50 - np.random.rand(50).cumsum(), 'k--')

plt.show()
fig = plt.figure(figsize=(10, 8), facecolor='gray')



ax1 = fig.add_subplot(2, 2, 1) # 返回 Axes 对象

ax1.plot(np.random.rand(50).cumsum(), 'k--')

ax1.plot(np.random.rand(50).cumsum(), 'b--')



ax2 = fig.add_subplot(2, 2, 2)

ax2.hist(np.random.rand(50), alpha=0.5)



ax4 = fig.add_subplot(2, 2, 4)

df = pd.DataFrame(np.random.rand(10, 4), columns=list('abcd'))

ax4.plot(df, '--.', alpha=.5)

plt.show()
fig, axes = plt.subplots(2, 3, figsize=(10, 4))



s = pd.Series(np.random.randn(1000).cumsum())

# 索引，第二个子图

ax1 = axes[0, 1]

ax1.plot(s)

plt.show()
# 参数调整，共用坐标轴

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)

for i in range(2):

    for j in range(2):

        axes[i, j].hist(np.random.randn(500), color='k', alpha=0.6)

# wspace, hspace: 控制宽度和高度的百分比，比如子图之间的间距

plt.subplots_adjust(wspace=0, hspace=0)

plt.show()
df = pd.DataFrame(np.random.randn(1000,4), index=s.index, columns=list('ABCD'))

df = df.cumsum()

df.plot(style='--.', alpha=0.4, grid=True, figsize=(20, 8), subplots=True,

        layout=(1,4), sharex=False)

plt.subplots_adjust(wspace=0.1, hspace=0)

plt.show()
# pandas 时间序列

s = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))

s = s.cumsum()

s.plot(kind='line', label='what', style='--.b', alpha=0.3, use_index=True, rot=45,

       grid=True, ylim=[-50, 50], yticks=list(range(-50, 50, 10)), figsize=(8, 4),

       title='Title', legend=True)

plt.grid(True, linestyle='--', color='green', linewidth=2, axis='x')

# plt.grid(True, linestyle='-', color='red', linewidth='0.5', axis='y')

plt.show()
df = pd.DataFrame(np.random.randn(1000, 4), index=s.index, columns=list('ABCD')).cumsum()

df.plot(kind='line', style='--.', alpha=0.4, use_index=True, rot=45, grid=True,

        figsize=(8, 4), title='test', legend=True, subplots=True, colormap='Greens')

plt.show()
fig, axes = plt.subplots(4, 1, figsize=(10, 10))



s = pd.Series(np.random.randint(0, 10, 16), index=list('abcdefghijklmnop'))

df = pd.DataFrame(np.random.rand(10, 3), columns=['a', 'b', 'c'])



# 柱状图方法一：

s.plot(kind='bar', color='k', grid=True, alpha=0.5, ax=axes[1]) # ax 选择子图位置



# 多柱状图

df.plot(kind='bar', ax=axes[0], grid=True, colormap='Reds_r')



# 多堆叠图

df.plot(kind='bar', ax=axes[2], grid=True, colormap='Blues_r', stacked=True)



# 柱状图方法二：

df.plot.barh(ax=axes[3], grid=True, stacked=True, cmap='BuGn_r')

plt.show()
plt.figure(figsize=(10, 6))



x = np.arange(10)

y1 = np.random.rand(10)

y2 = -np.random.rand(10)



plt.bar(x, y1, width=1, facecolor='yellowgreen', edgecolor='white')

plt.bar(x, y2, width=1, facecolor='lightskyblue', edgecolor='white', yerr=y2*0.1)



# zip 将可迭代对象打包成元组

for i,j in zip(x, y1):

    plt.text(i-0.2, j-0.15, '%.2f' % j, color='k')

for i,j in zip(x, y2):

    plt.text(i-0.2, j-0.05, '%.2f' % -j, color='r')

plt.show()
fig, axes = plt.subplots(2, 1, figsize=(8, 6))



df1 = pd.DataFrame(np.random.rand(10, 4), columns=list('abcd'))

df2 = pd.DataFrame(np.random.rand(10, 4), columns=list('abcd'))



df1.plot.area(cmap='Greens_r', alpha=0.5, ax=axes[0])

df2.plot.area(stacked=True, cmap='Set2', alpha=0.5, ax=axes[1])

plt.show()
fig, axes = plt.subplots(2, 1, figsize=(8, 6))



x= np.linspace(0, 1, 500)

y1 = np.sin(4 * np.pi * x) * np.exp(-5 * x)

y2 = -np.sin(4 * np.pi * x) * np.exp(-5 * x)



axes[0].fill(x, y1, 'r', alpha=0.5, label='y1')

axes[0].fill(x, y2, 'g', alpha=0.5, label='y2')

# 同效果

# plt.fill(x, y1, 'r', x, y2, 'g', alpha=0.5)



x = np.linspace(0, 5 * np.pi, 1000)

y1 = np.sin(x)

y2 = np.sin(2 * x)

# 填充两个函数之间的区域

axes[1].fill_between(x, y1, y2, color='b', alpha=0.5, label='area')



for ax in axes:

    ax.legend()

    ax.grid()



plt.show()
# np.random.seed(1)

%time

s = pd.Series(3 * np.random.rand(4), index=['a', 'b', 'c', 'd'], name='series')

fig = plt.figure(figsize=(10, 8))

plt.subplot(1, 2, 1)

plt.axis('equal') # 保证长宽相等,画出来才圆

plt.pie(s, explode=[0.05, 0, 0, 0], labels=s.index, colors=['r', 'g', 'b', 'c'],

        autopct='%.2f%%', pctdistance=0.5, labeldistance=1.2, shadow=True,

        startangle=0, radius=1.5)



plt.subplot(1,2,2)

plt.axis('equal') # 保证长宽相等,画出来才圆

plt.pie(s, explode=[0.05, 0, 0, 0], labels=s.index, colors=['r', 'g', 'b', 'c'],

        autopct='%.2f%%', pctdistance=0.6, labeldistance=1.2, shadow=False,

        startangle=0, radius=1.5)

plt.subplots_adjust(wspace=0.5, hspace=0)

plt.show()
# 直方图

s = pd.Series(np.random.randn(1000))

s.hist(bins=20, histtype='bar', align='mid', orientation='vertical', alpha=0.5,

       grid=False, density=True, label='hist')

# 密度图

s.plot(kind='kde', style='k--', label='line')

plt.legend(loc='upper left') # 就只有几个固定值

plt.show()
# 堆叠直方图

plt.figure(figsize=(10, 6))

df = pd.DataFrame({'a': np.random.randn(1000) + 1, 'b': np.random.randn(1000),

                   'c': np.random.randn(1000) - 1, 'd': np.random.randn(1000) - 2},

                  columns=list('abcd'))

df.plot.hist(bins=20, cmap='Greens_r', alpha=0.5, grid=True, stacked=True)



# 生成多个直方图

df.hist(bins=20)

plt.subplots_adjust(wspace=0.3, hspace=0.5)

plt.show()
plt.figure(figsize=(8, 6))



x = np.random.randn(1000)

y = np.random.randn(1000)



plt.scatter(x, y, marker='o', s=np.random.rand(1000)*100, cmap='Reds_r', c=y, alpha=0.8)

plt.show()
df = pd.DataFrame(np.random.randn(100, 4), columns=list('abcd'))

pd.plotting.scatter_matrix(df, figsize=(10,6), marker='o', diagonal='kde', alpha=0.5, 

                  range_padding=0.5)

plt.show()
# df.plot.box() 绘制

fig, axes = plt.subplots(2, 1, figsize=(10, 6))

df = pd.DataFrame(np.random.rand(10, 5), columns=list('ABCDE'))

# 箱型图颜色

# boxes → 箱线

# whiskers → 分位数与 error bar 横线之间竖线的颜色

# medians → 中位数线颜色

# caps → error bar 横线颜色

color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='blue')



df.plot.box(ylim=[0, 1.2], grid=True, color=color, ax=axes[0])

df.plot.box(vert=False, positions=[1,4,5,6,8], grid=True, color=color, ax=axes[1])

plt.show()
df = pd.DataFrame(np.random.rand(10, 5), columns=list('ABCDE'))

plt.figure(figsize=(10, 4))



f = df.boxplot(sym='+', vert=True, whis=1.5, patch_artist=True, meanline=False,

               showmeans=True, showbox=True, showcaps=True, showfliers=True,

               notch=False, return_type='dict')

plt.title('boxplot')

for box in f['boxes']:

    box.set(color='b', linewidth=2)

    box.set(facecolor='y', alpha=0.5)

for whisker in f['whiskers']:

    whisker.set(color='k', linewidth=0.5, linestyle='--')

for cap in f['caps']:

    cap.set(color='green', linewidth=2)

for median in f['medians']:

    median.set(color='r', linewidth=2)

for flier in f['fliers']:

    flier.set(marker='+', color='yellow', alpha=0.5)

plt.show()
# 分组绘制

df = pd.DataFrame(np.random.rand(10, 2), columns=['Col1', 'Col2'])

df['X'] = pd.Series(['A','A','A','A','A','B','B','B','B','B'])

df['Y'] = pd.Series(['A','B','A','B','A','B','A','B','A','B'])



df.boxplot(by='X')

df.boxplot(column=['Col1', 'Col2'], by=['X', 'Y'])



plt.show()
plt.figure(figsize=(12, 8))

x = np.array([[32, 24], [22, 18]])

plt.imshow(x)

plt.colorbar()

plt.text(1, 0, 'h', color='r', fontsize=20)

plt.text(1, 0, 'O', color='k', fontsize=20)

plt.show()
fig, axes = plt.subplots(1, 2, figsize=(10, 8))

x = np.array([[32, 24], [56, 24], [22, 18]])

for ax in axes:

    im = ax.imshow(x,interpolation='nearest')

    ax.figure.colorbar(im, ax=ax)

    for i in range(x.shape[0]):

        for j in range(x.shape[1]):

            ax.text(j, i, x[i][j], color='r') # x, y 对应列，行

plt.show()
plt.figure(figsize=(12, 6))



x=np.linspace(-3,3,300)

y=np.linspace(-3,3,300)

X, Y = np.meshgrid(x, y)



Z = 2 * X - 3 * Y

plt.subplot(1, 4, 1)

plt.contour(X, Y, Z, cmap=plt.cm.Set1)



Z = X ** 2 + Y ** 2

plt.subplot(1, 4, 2)

plt.contourf(X, Y, Z, colors=['red', 'blue', 'green'], origin='upper')



Z = (4 * X - 2 * Y) > 0

Z = Z.astype(np.uint8)

plt.subplot(1, 4, 3)

plt.contourf(X, Y, Z, cmap=plt.cm.Blues, origin='lower')



plt.subplot(1, 4, 4)

plt.contourf(X, Y, Z, cmap=plt.cm.Blues, origin='upper')





plt.show()