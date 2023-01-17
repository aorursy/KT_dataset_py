from IPython.display import Image

Image("Videos//Capture.PNG")
import matplotlib as mpt

import matplotlib.pyplot as plt

#%matplotlib inline

#%matplotlib notebook

mpt.get_backend()
plt.plot(2,4)
plt.plot(2,4,'*')
plt.figure()

plt.plot(4,3,'*')

ax = plt.gca()

ax.axis([0,5,0,10])
ax.get_children()
plt.figure()

plt.plot(4,3,'*')

plt.plot(7,2,'*')

plt.plot(2,7,'*')
plt.figure()

plt.plot(4,3,'*')

plt.plot(7,2,'*')

plt.plot(2,7,'*')

ax = plt.gca()

ax.get_children()
plt.plot([1, 2, 3, 4,5],'*')

plt.ylabel('numbers')

plt.show()
plt.plot([1, 2, 3, 4,5])

plt.ylabel('numbers')

plt.show()
plt.plot([1, 2, 3, 4], [1, 7, 9, 19], 'yo')

plt.axis([0, 5, 0, 30])

plt.show()
import numpy as np
a = np.array([1,2,3,4,5])

b = a

plt.figure()

plt.scatter(a,b)
a = np.array([1,2,3,4,5])

b = a

#color = ['blue']*(len(a)-1)

#color.append('red')

color=['blue','red','blue','red','blue']

plt.figure()

plt.scatter(a,b, s = 200, c = color)
zip_generator = zip([1,3,5,7],[2,4,6,8])

myzip=list(zip_generator)

myzip
zip_generator = zip([1,3,5,7],[2,4,6,8])

a, b = zip(*zip_generator)

print(a)

print(b)
plt.figure()

plt.scatter(a[:2],b[:2], s = 200, c = 'red', label= 'data1')

plt.scatter(a[2:],b[2:], s = 200, c = 'yellow', label= 'data2')

plt.figure()

plt.scatter(a[:2],b[:2], s = 200, c = 'red', label= 'data1')

plt.scatter(a[2:],b[2:], s = 200, c = 'yellow', label= 'data2')

plt.xlabel('the data on x')

plt.ylabel('the data on y')

plt.title('data')
plt.figure()

plt.scatter(a[:2],b[:2], s = 200, c = 'red', label= 'data1')

plt.scatter(a[2:],b[2:], s = 200, c = 'yellow', label= 'data2')

plt.xlabel('the data on x')

plt.ylabel('the data on y')

plt.title('data')

#plt.legend()

plt.legend(loc=4, frameon=False, title='legend' )
plt.gca().get_children()
lineardata = np.array([1,2,3,4,5])

qubedata = lineardata**3

plt.figure()

plt.plot(lineardata, '-o', qubedata, '-o')

#plt.plot([10,20,30,40], '--b')

#plt.xlabel('number of points')

#plt.ylabel('values')

#plt.title('the plot')

#plt.legend(['series1', 'series2', 'series3'])
plt.gca().fill_between(range(len(lineardata)), lineardata, qubedata, facecolor='red', alpha=0.50)
plt.figure()

xval = range(len(lineardata))

plt.bar(xval, lineardata, width=0.5)
xval
newx = []

for f in xval:

    newx.append(f+0.5)

plt.bar(newx, qubedata, width=0.5, color = 'red')
from random import randint
l_err = [randint(0,10) for g in range(len(lineardata))]

plt.bar(xval, lineardata, width=0.5, color = 'red', yerr= l_err)
plt.figure()

xval = range(len(lineardata))

plt.bar(xval, lineardata, width=0.5, color='b')

plt.bar(xval, lineardata, width=0.5, bottom=lineardata, color='r')
plt.figure()

xval = range(len(lineardata))

plt.barh(xval, lineardata, height=0.5, color='b')

plt.barh(xval, lineardata, height=0.5, left=lineardata, color='r')
#subplots

plt.subplot(1,2,1)

plt.plot(lineardata, '-o')

plt.subplot(1,2,2)

plt.plot(qubedata, '-o')

#plt.subplot(1,2,1)

#plt.plot(qubedata, '-x')
plt.figure()

x1 = plt.subplot(1,2,1)

plt.plot(lineardata, '-o')

x2 = plt.subplot(1,2,2, sharey = x1)

plt.plot(qubedata, '-*')

fig, ((ax1,ax2,ax3), (ax3,ax4,ax5), (ax6,ax7,ax8))= plt.subplots(3,3,sharex=True, sharey=False)

ax7.plot(lineardata, '--x')
#histograms

fig, ((ax1,ax2), (ax3,ax4))= plt.subplots(2,2,sharex=True, sharey=False)

ax = [ax1, ax2, ax3, ax4]



for n in range(0, len(ax)):

    sample_size = 10**(n+1)

    sample = np.random.normal(loc=0.0, scale=1.0, size= sample_size)

    ax[n].hist(sample)

    ax[n].set_title('n={}'.format(sample_size))



      

fig, ((ax1,ax2), (ax3,ax4))= plt.subplots(2,2,sharex=True, sharey=False)

ax = [ax1, ax2, ax3, ax4]



for n in range(0, len(ax)):

    sample_size = 10**(n+1)

    sample = np.random.normal(loc=0.0, scale=1.0, size= sample_size)

    ax[n].hist(sample, bins=100)

    ax[n].set_title('n={}'.format(sample_size))
plt.figure()

sample = np.random.normal(loc=0.0, scale=1.0, size= 1000)

sample1 = np.random.random(size= 1000)

plt.scatter(sample, sample1)
#heatmap

plt.figure()

sample = np.random.normal(loc=0.0, scale=1.0, size= 1000)

sample1 = np.random.random(size= 1000)

_=plt.hist2d(sample, sample1, bins=100)
#GridSpec



import matplotlib.gridspec as gridspec
fig2 = plt.figure()

spec2 = gridspec.GridSpec(ncols=2, nrows=3, figure=fig2)

f2_ax1 = fig2.add_subplot(spec2[0, 0])

f2_ax2 = fig2.add_subplot(spec2[0, 1])



f2_ax3 = fig2.add_subplot(spec2[1, 0])

f2_ax4 = fig2.add_subplot(spec2[1, 1])

f2_ax3 = fig2.add_subplot(spec2[2, 0])

f2_ax4 = fig2.add_subplot(spec2[2, 1])
#fig1, f1_axes = plt.subplots(ncols=2, nrows=3, constrained_layout=True)
#The power of gridspec comes in being able to create subplots that span rows and columns

fig3 = plt.figure(constrained_layout=True)

gs = fig3.add_gridspec(3, 3)

f3_ax1 = fig3.add_subplot(gs[0, :])

f3_ax1.set_title('gs[0, :]')

f3_ax2 = fig3.add_subplot(gs[1, :-1])

f3_ax2.set_title('gs[1, :-1]')

f3_ax3 = fig3.add_subplot(gs[1:, -1])

f3_ax3.set_title('gs[1:, -1]')

f3_ax4 = fig3.add_subplot(gs[-1, 0])

f3_ax4.set_title('gs[-1, 0]')

f3_ax5 = fig3.add_subplot(gs[-1, -2])

f3_ax5.set_title('gs[-1, -2]')
import matplotlib.pyplot as plt

#plt.plot([1,2,3,4])

plt.plot([1,2,3,4],[1,4,9,16],'ro')

plt.title('My first plot')

plt.plot()

plt.plot([1,2,4,2,1,0,1,2,1,4],linewidth=2.0)
# Horizontal Subplots

import numpy as np

t = np.arange(0,5,0.1)

y1 = np.sin(2*np.pi*t)

y2 = np.sin(2*np.pi*t)

plt.subplot(211)

plt.plot(t,y1,'b-.')

plt.subplot(212)

plt.plot(t,y2,'r--')

# Verical Slubplots 121 and 122
plt.axis([0,5,0,20])

plt.title('My first plot')

# plt.title('My first plot',fontsize=20,fontname='Times New Roman')

plt.xlabel('Counting')

plt.ylabel('Square values')

plt.plot([1,2,3,4],[1,4,9,16],'ro')

plt.text(1,1.5,'First')

plt.text(2,4.5,'Second')

plt.legend(['First series'])

# The position of the legend can be changed based on the loc paramtere

# Some values of loc are 
import matplotlib.pyplot as plt

import numpy as np

x = np.arange(-2*np.pi,2*np.pi,0.01)

y = np.sin(3*x)/x

y2 = np.sin(2*x)/x

y3 = np.sin(4*x)/x

plt.plot(x,y)

plt.plot(x,y2)

plt.plot(x,y3)
# Histogram

pop = np.random.randint(0,100,100)

n,bins,patches = plt.hist(pop,bins=20)



# Bar Chart

index = np.arange(5)

values1 = [5,7,3,4,6]

plt.bar(index,values1)

plt.xticks(index+0.4,['A','B','C','D','E'])
# Scatter plot

rng = np.random.RandomState(0)

x = rng.randn(100)

y = rng.randn(100)

colors = rng.rand(100)

sizes = 500 * rng.rand(100)



plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,

            cmap='viridis')

plt.colorbar();  # show color scale
# Import dataset

import pandas as pd

titan=pd.read_csv('../input/train.csv')



# Assignment 1 Draw the Age distribution of titanic passengers Histogram

pop =titan['Age']

plt.title('Titanic Passenger Age')

n,bins,patches = plt.hist(pop,bins=5)





# Assignment 2 Draw the survived and not survived age Histogram

ns_age= titan['Age'][titan['Survived']==0].mean()

s_age = titan['Age'][titan['Survived']==1].mean()

index = np.arange(2)

values1 = [ns_age,s_age]

plt.bar(index,values1)

plt.xticks(index+0.4,['NS','S'])