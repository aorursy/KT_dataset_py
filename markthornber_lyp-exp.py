import matplotlib.pyplot as plt
import numpy as np
def makeaxes(axes,xmin,xmax,ymin,ymax):
    axes.spines['left'].set_position(('data', 0.0))
    axes.spines['bottom'].set_position(('data', 0.0))
    axes.spines['right'].set_color('none')
    axes.spines['top'].set_color('none')
    axes.set_xlim(xmin,xmax)
    axes.set_ylim(ymin,ymax)
# Define functions
x = np.arange(-3, 3 , 0.1) 
y1 = x**2 
y2 = x**3-4*x
# Do the formatting
fig, ax = plt.subplots()
makeaxes(ax,-3,3,-10,10)
plt.xlabel('x',horizontalalignment='right', x=1.0)
plt.ylabel('y',rotation=0,horizontalalignment='right',y=1.0)
plt.title('Polynomial Graphs',y = 1.05)
# Now plot
ax.plot(x,y1,label='$y=x^2$',color='red')
ax.plot(x,y2,label='$y=x^3-4x$',color='blue')
plt.legend(loc=(0.7,0))
plt.show()
from ipywidgets import interact

# Define a graph function
def plotgraph(a=1,b=0,c=0):
    x = np.arange(-3,3,0.1)
    y = a*x**2+b*x+c
    fig, ax = plt.subplots()
    makeaxes(ax,-3,3,-10,10)
    plt.xlabel('x',horizontalalignment='right', x=1.0)
    plt.ylabel('y',rotation=0,horizontalalignment='right',y=1.0)
    plt.title('$y={}x^2+{}x+{}$'.format(a,b,c),y = 1.05)
    plt.plot(x,y)
    
# Use floats to define intervals for the sliders or we get integer sliders
# Ending with a semicolon suppresses a useless output statement
interact(plotgraph,a=(-3.0,3.0),b=(-3.0,3.0),c=(-3.0,3.0));