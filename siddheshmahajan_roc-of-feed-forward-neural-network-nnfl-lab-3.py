from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
x1 = np.linspace(0,3,100) 
x2 = np.linspace(0,9,100) 
X1,X2 = np.meshgrid(x1,x2) 
net_1 = X1 - 1
net_2 = -X1 + 2
net_3 = X2
net_4 = -X2 + 3
o_1 = np.sign(net_1)
o_2 = np.sign(net_2)
o_3 = np.sign(net_3)
o_4 = np.sign(net_4)
net_5 = (o_1+o_2+o_3+o_4)-3.5

o_5 = np.sign(net_5)
fig = plt.figure(figsize=(16,10)) 
ax = plt.axes(projection = '3d') 
ax.plot_surface(X1,X2,o_5) 
ax.set_xlabel('X1') 
ax.set_ylabel('X2') 
ax.set_zlabel('Output 5') 
ax.set_title('Region of convergence')
