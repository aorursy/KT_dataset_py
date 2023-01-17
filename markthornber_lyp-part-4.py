import math

print(math.pi)
print(round(math.pi,3))
for i in range(10):
    x = round(i*math.pi/10,3)
    y = round(math.sin(x),3)
    print(x,y)
x = math.cos(math.radians(60))
print(round(x,1))
x = 3.25789
y = 2.841598
print('The coordinates are({},{})'.format(x,y))
print('Here is x to 2dp--{:.2f}'.format(x))
print('Here is x to 3dp--{:.3f}'.format(x))
print('Here is x with total width 6 and 2dp--{:6.2f}'.format(x))
import math as m
print('  x  | cos(x) ')
print('--------------')  
for i in range(11):
    x = i*m.pi/10
    y = m.cos(x)
    print('{:.2f} | {:5.2f}'.format(x,y))
import matplotlib.pyplot as plt
plt.plot([1,2,3,4],[5,7,4,8])
plt.show()
x = [i*m.pi/10 for i in range(21)]
y = [m.cos(i*m.pi/10) for i in range(21)]
plt.plot(x,y,label='y=cos(x)')
plt.legend()
plt.show()
x = [i*m.pi/10 for i in range(21)]
y1 = [m.cos(i*m.pi/10) for i in range(21)]
y2 = [m.sin(i*m.pi/10) for i in range(21)]
plt.plot(x,y1,label='y=cos(x)')
plt.plot(x,y2,label='y=sin(x)')
plt.xlabel('x in radians')
plt.ylabel('y')
plt.title('Graphs of sin, cos and tan')
plt.legend()
plt.show()
x = [i*m.pi/10 for i in range(21)]
y1 = [m.cos(i*m.pi/10) for i in range(21)]
y2 = [m.sin(i*m.pi/10) for i in range(21)]
# Here is the key command
fig, ax = plt.subplots()
# This sets the axis positions to be centred where the data is equal to 0.0
ax.spines['left'].set_position(('data', 0.0))
ax.spines['bottom'].set_position(('data', 0.0))
# This hides the rest of the outside box
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# We plot to the axes (this allows us to superimpose graphs with different axes)
ax.plot(x,y1,label='y=cos(x)')
ax.plot(x,y2,label='y=sin(x)')
# Specify the box limits
plt.xlim(-1,7)
plt.ylim(-1.5,1.5)
# Add Labels and position at the ends of the axes
plt.xlabel('x in radians',horizontalalignment='right', x=1.0)
plt.ylabel('y', rotation=0, horizontalalignment='right', y=1.0)
plt.title('Graphs of sin and cos')
# This locates the legend 3/20 of the way in and 1/10 of the way up
plt.legend(loc=(0.15,0.1))
plt.show()

def makeaxes(axes,xmin,xmax,ymin,ymax):
    axes.spines['left'].set_position(('data', 0.0))
    axes.spines['bottom'].set_position(('data', 0.0))
    axes.spines['right'].set_color('none')
    axes.spines['top'].set_color('none')
    axes.set_xlim(xmin,xmax)
    axes.set_ylim(ymin,ymax)

import numpy as np
# This creates a list of points to plot, spaced 0.1 apart
x = np.arange(0, 3 * np.pi, 0.1) 
# Now apply numpy functions to our list
y1 = np.cos(x) 
y2 = np.sin(x)
# Now the formatting stuff
fig, ax = plt.subplots()
makeaxes(ax,-0.5,3*np.pi,-1.5,1.5)
# Make the ticks nice
plt.xticks(np.arange(0, 3 * np.pi+0.5, np.pi/2),['0','$\pi$/2','$\pi$','$3\pi$/2','$2\pi$','$5\pi$/2','$3\pi$'] )
# Add Labels and position at the ends of the axes
plt.xlabel('x in radians',horizontalalignment='right', x=1.0)
plt.ylabel('y', rotation=0, horizontalalignment='right', y=1.0)
plt.title('Graphs of sin and cos')
# Now plot
ax.plot(x,y1,label='y=cos(x)',color='red')
ax.plot(x,y2,label='y=sin(x)',color='purple')
# This locates the legend 1/10 of the way in and at the bottom
plt.legend(loc=(0.1,0))
plt.show()
x = np.arange(-2, 2, 0.1) 
# Now the formatting stuff
fig, ax = plt.subplots()
makeaxes(ax,-2,2,-3,3)
plt.xlabel('x',horizontalalignment='right', x=1.0)
plt.ylabel('y', rotation=0, horizontalalignment='right', y=0.95)
plt.title('Graphs of $y=ln(x^2+c)$ for $0\leq c<\leq 3$')
# Calculate y values and plot
for c in np.arange(0,3,0.1):
    y = np.log(x**2+c)
    ax.plot(x,y, color=(c/3,0,1-c/3))

plt.show()
# This line changes the maths font to Computer Modern
plt.rc('mathtext', fontset="cm")
# This sets up an array of graphs. The variable "axes" is an array of the axes for each subgraph
fig, axes = plt.subplots(nrows=4, ncols=5,figsize=(15,15))
# Add a main title
fig.suptitle('Cross Sections of $z=x^3-3xy^2$',fontsize=18, y=1.03)
# We use the same x values for every graph
x = np.arange(-2,2,0.1)
# Now use nested loops to set up the graphs
for row in range(4):
    for col in range(5):     
        y = row-2+col/5
        # Numpy lists can be used in arithmetic
        z = x**3-3*x*y**2
        # extract the correct set of axes and use it to plot
        subplt=axes[row,col]
        makeaxes(subplt,-2,2,-8,8)
        subplt.set_xlabel('$x$',x=1.0)
        subplt.set_ylabel('$z$',y=1.0, rotation=0)
        subplt.set_title('$y={:.1f}$'.format(y), loc='right')
        subplt.plot(x,z)
        
fig.tight_layout()
plt.show()

    