import matplotlib.pyplot as plt
plt.plot([1,2,3,4,5],[4,6,3,8,12])
plt.show()
from matplotlib import pyplot as plt
from matplotlib import style
style.use("ggplot")
x1,x2,y1,y2=[1,4,9,12,15],[2,7,10,13,17],[22,34,12,33,28],[29,22,24,26,23]
plt.plot(x1,y1,'m',label='line 1',linewidth=2)
plt.plot(x2,y2,'red',label='line 2',linewidth=3)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True,color='g')
plt.show()
plt.step([1,3,5,7,9],[1,9,25,49,81],color='red',linewidth=5)
plt.grid(True,color='aquamarine')
plt.show()
import numpy as np
x=np.arange(0,10,0.1)
plt.plot(x,np.tan(x),color='aqua')
plt.grid(True,color='k')
plt.show()
def year_wise_IPL(pos,team,col):
    plt.bar(list(range(2008,2020)),pos,label=team,color=col)
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('Performance')
    plt.title(team+' Year-Wise Performance')
    plt.show()
teams=['MI','CSK','RCB']
pos=[[4,2,7,6,5,8,5,8,4,8,4,8],[7,6,8,8,7,7,6,7,0,0,8,7],[2,7,6,7,4,4,2,6,7,1,3,1]]
c=['#1589FF','#FFD801','red']
for i in range(3):
    year_wise_IPL(pos[i],teams[i],c[i])