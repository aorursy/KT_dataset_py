import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
x = np.linspace(1,10,10)
x
y = x**2
y
plt.plot(x,y) 
plt.xlabel('Squares')
plt.ylabel('Numbers')
plt.title('Exponential')
# or 
#plt.plot(x,y, 'r')
plt.subplot(1,2,1)
plt.plot(x,y)
plt.subplot(1,2,2)
plt.plot(y,x,'g')
fig = plt.figure()
axes = fig.add_axes([0,0,0.9,0.9]) #([from left, from bottom, width, height])
axes.plot(x,y)
fig = plt.figure()
axes = fig.add_axes([0,0,0.9,0.9]) #([from left, from bottom, width, height])
axes2 = fig.add_axes([0.1,0.3,0.5,0.5])
axes.plot(x,y)
axes2.plot(x,y)
axes.set_title('Larger Plot')
axes2.set_title('Smaller Plot')
fig, axes = plt.subplots(nrows = 3,ncols = 3)
plt.tight_layout()
fig ,axes = plt.subplots(nrows = 1, ncols = 2)
print(axes)

for ax in axes:
    ax.plot(x,y)

fig ,axes = plt.subplots(nrows = 1, ncols = 2)
axes[0].plot(x,y)
axes[1].plot(y,x)
axes[0].set_title('Plot 1')
axes[1].set_title('Plot 2')
plt.figure(figsize = (8,2))
plt.plot(x,y)
fig, axes = plt.subplots(nrows = 2, ncols = 1,figsize = (8,3))
axes[0].plot(x,y)
axes[1].plot(y,x)
plt.tight_layout()
axes
fig.savefig('squeeze.png')
fig = plt.figure()
axes = fig.add_axes([0,0,0.9,0.9])
axes.plot(x,x**3,'-r', label = 'X Squared')
axes.plot(x,x**2,'-b', label = 'X Cubed')

axes.legend(loc = 0)
fig = plt.figure()
axes = fig.add_axes([0,0,0.9,0.9])
axes.plot(x,x**3,'-r', label = 'X Squared')
axes.plot(x,x**2,'-b', label = 'X Cubed')

axes.legend(loc = (0.1,0.2))
fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
axes.plot(x,y, color = 'purple',alpha = 0.5, linestyle = '-.', linewidth = 3) # linewidth or lw
fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (6,6))
axes[0][0].plot(x,y, color = 'green', linestyle = '-.')
axes[0][1].plot(x,y, color = 'purple', linestyle = ':')
axes[1][0].plot(x,y, color = '#3399FF', linestyle = '-')
axes[1][1].plot(x,y, color = '#FF007F', linestyle = 'steps')
axes[2][0].plot(x,y, color = '#CCCC00', linestyle = 'dotted')
axes[2][1].plot(x,y, color = '#808080', linestyle = '-')
fig = plt.figure()
axes = fig.add_axes([0,0,1,1])

axes.plot(x,y, color = 'black', ls = '--',lw = 4, marker = 'o', markersize = 20, markerfacecolor = 'pink'
         ,markeredgewidth = 4, markeredgecolor = 'red')

axes.plot(x,x*1.1, color = 'blue', ls = '--',lw = 3, marker = '+', markersize = 20, markeredgecolor = 'green')
x = np.linspace(0,50,50)
y= x**2
fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
axes.plot(x,y, color = 'purple') # linewidth or lw

axes.set_xlim(0,20)
axes.set_ylim(0,1500)
