# First of all we import our modules 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pylab import *
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import itertools
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# Any results you write to the current directory are saved as output.
#We import our data and investigate
df = pd.read_csv('../input/kaggle.txt',delimiter="\t")
df = df.iloc[:,:6] # We use only first 5 columns
df.head()
#We are starting create our graphic
matplotlib.rcParams.update({'font.size':10})
fig=plt.figure()

ax=fig.gca(projection='3d')

for i in range(1,6):
    
    z=df.iloc[:, i]
    
    y=np.array(sorted(list(range(1,len(z)+1))*2)).reshape((48,1))  #this is hours
    a=np.array([i,i+1]*len(z)).reshape((48,1))                     #this is stations line width
    
    points = np.concatenate((a, y), axis=1)        #this is base of the graphic
    b =list(itertools.chain(*zip(z,z)))            # Z Dimension
    b = np.array(b).reshape((48,1))       #for 48 data in 1D
   
    yi  =np.linspace(min(a),max(a))         

    xi  =np.linspace(min(y),max(y))

    X,Y =np.meshgrid(yi,xi)    # meshgrid of the graphic

    Z = griddata(points,b,(yi,xi),method="linear")  # This is interpolation technique, you could change method as "cubic" or "nearest" 
    #griddata use only 1D arrays, IMPORTANT
    ax.plot_surface(Y,X,Z,rstride=1,cstride=5)   

    ax.set_zlim3d(0,50)   # you can change the limits of Z
ax.grid(False)
ax.w_xaxis.pane.set_visible(False)
ax.set_title('Hourly Total Precipitations')
plt.yticks(rotation=330)
ax.set_yticks([1.5,2.5,3.5,4.5,5.5])
ax.set_yticklabels(['Sta1','Sta2','Sta3','Sta4',"Sta5"])
plt.xticks(rotation=50)
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
ax.set_xticklabels(["21","","23","","1","","3","",'5','','7','','9',"",'11','','13',"",'15','','17','','19',''])
ax.set_xlabel('Hours')
ax.set_zlabel('Precipitation Totals (mm)')
show()
matplotlib.rcParams.update({'font.size':10})
fig=plt.figure()

ax=fig.gca(projection='3d')

for i in range(1,6):
    
    z=df.iloc[:, i]
    
    y=np.array(sorted(list(range(1,len(z)+1))*2)).reshape((48,1))  #this is hours
    a=np.array([i,i+1]*len(z)).reshape((48,1))                     #this is stations line width
    
    points = np.concatenate((a, y), axis=1)        #this is base of the graphic
    b =list(itertools.chain(*zip(z,z)))            # Z Dimension
    b = np.array(b).reshape((48,1))       #for 48 data in 1D
   
    yi  =np.linspace(min(a),max(a))         

    xi  =np.linspace(min(y),max(y))

    X,Y =np.meshgrid(yi,xi)    # meshgrid of the graphic

    Z = griddata(points,b,(yi,xi),method="linear")  # This is interpolation technique, you could change method as "cubic" or "nearest" 
    #griddata use only 1D arrays, IMPORTANT
    surf = ax.plot_surface(Y,X,Z,rstride=1,cstride=5,cmap=cm.jet)   

    ax.set_zlim3d(0,50)   # you can change the limits of Z
ax.grid(False)
ax.w_xaxis.pane.set_visible(False)
ax.set_title('Hourly Total Precipitations')
plt.yticks(rotation=330)
ax.set_yticks([1.5,2.5,3.5,4.5,5.5])
ax.set_yticklabels(['Sta1','Sta2','Sta3','Sta4',"Sta5"])
plt.xticks(rotation=50)
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
ax.set_xticklabels(["21","","23","","1","","3","",'5','','7','','9',"",'11','','13',"",'15','','17','','19',''])
ax.set_xlabel('Hours')
ax.set_zlabel('Precipitation Totals (mm)')
fig.colorbar(surf, shrink=0.5, aspect=5)
show()
