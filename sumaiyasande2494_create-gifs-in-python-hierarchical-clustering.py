import numpy as np

import pandas as pd

from numpy.random import default_rng

from scipy.cluster import hierarchy

import matplotlib.pyplot as plt

!pip install celluloid

from celluloid import Camera

import seaborn as sns

from matplotlib.animation import FuncAnimation
rng = default_rng(5003)

# generating data (that I know comes from two groups)

X1 = rng.normal(loc=2.0, scale=0.5, size=20).reshape((10,2))

X2 = rng.normal(loc=-2.0, scale=0.5, size=20).reshape((10,2))

X3 = rng.normal(loc=-5.0, scale=0.5, size=20).reshape((10,2))

X4 = rng.normal(loc=5.0, scale=0.5, size=20).reshape((10,2))

X5 = rng.normal(loc=0.0, scale=0.5, size=20).reshape((10,2))



dummy_data = np.vstack([X1, X2, X3, X4, X5])

dummy_data.shape
Z = hierarchy.linkage(dummy_data)

grps = hierarchy.cut_tree(Z,5).ravel()

colors = np.array(['b','g','r','m','k','y','c'])

fig=plt.figure()

ax=fig.add_axes([0,0,1,1])

ax.scatter(dummy_data[:,0], dummy_data[:, 1], facecolor=colors[grps]); 

ax.set_title('Clustering Algo Output');
#credit : https://stackoverflow.com/questions/44575681/how-do-i-encircle-different-data-sets-in-scatter-plot

def encircle(x,y, ax=None, **kw):

    if not ax: ax=plt.gca()

    p = np.c_[x,y]

    mean = np.mean(p, axis=0)

    d = p-mean

    r = np.max(np.sqrt(d[:,0]**2+d[:,1]**2 ))

    circ = plt.Circle(mean, radius=1.05*r,**kw)

    ax.add_patch(circ)
#colors = np.array(['b','g','r','m','k','y','c'])

fig=plt.figure()

ax=fig.add_axes([0,0,1,1])

data = pd.DataFrame(dummy_data)

data.columns =['x','y']

grps = hierarchy.cut_tree(Z,5).ravel()

data['Group']=grps

ax.scatter(dummy_data[:,0], dummy_data[:, 1], facecolor=colors[data.Group]); 

ax.set_title('Clustering Algo Output');

encircle(data[data.Group==0].x,data[data.Group==0].y,ec="r", fc="gold", alpha=0.2)

encircle(data[data.Group==1].x,data[data.Group==1].y,ec="k", fc="red", alpha=0.2)

encircle(data[data.Group==2].x,data[data.Group==2].y,ec="k", fc="blue", alpha=0.2)

encircle(data[data.Group==3].x,data[data.Group==3].y,ec="k", fc="green", alpha=0.2)

encircle(data[data.Group==4].x,data[data.Group==4].y,ec="k", fc="gray", alpha=0.2)



plt.gca().relim()

plt.gca().autoscale_view()

plt.show()

from celluloid import Camera

colors = np.array(['b','g','r','m','k','y','c'])

data = pd.DataFrame(dummy_data)

data.columns =['x','y']

fig = plt.figure()

camera = Camera(fig)



for i in range(1,7):

    grps = hierarchy.cut_tree(Z,i).ravel()

    data['Group']=grps

    plt.scatter(data.x, data.y, facecolor=colors[data.Group])

    for j in range(len(data.Group.unique())):

        encircle(data[data.Group==j].x,data[data.Group==j].y,ec="k", fc=colors[j], alpha=0.1)

    camera.snap()



anim = camera.animate()

anim.save('animation1.gif',writer='imagemagick')
from matplotlib.animation import FuncAnimation

fig = plt.figure()

def animate(i):

    grps = hierarchy.cut_tree(Z,i).ravel()

    data['Group']=grps

    plt.scatter(data.x, data.y, facecolor=colors[data.Group])

    for j in range(len(data.Group.unique())):

        encircle(data[data.Group==j].x,data[data.Group==j].y,ec="k", fc=colors[j], alpha=0.1)

    

plt.legend()    

ani = FuncAnimation(fig, animate, frames=6, repeat=True)

anim.save('animation2.gif', writer='imagemagick')