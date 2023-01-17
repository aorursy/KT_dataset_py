import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import pandas as pd
import matplotlib.gridspec as gsp

%matplotlib notebook
# Monthly mean precipitation in mm in Kolkata,India  
prec = pd.read_excel('../input/prec.xlsx')
prec=prec.T
prec
# Monthly mean of Temperatures in Centigrades in Kolkata,India
Tavg0 = pd.read_excel('../input/avg_temp.xlsx')
Tavg=Tavg0.T
Tavg

# Monthly mean of Temperatures from 1991-2001 
me=Tavg0.describe()
me=me.T
me=me['mean']
me.reset_index()
me=me[1:]
me
Tavg=Tavg.rename(columns={0:1991,1:1992,2:1993,3:1994,4:1995,5:1996,6:1997,7:1998,8:1999,9:2000,10:2001})
Tavg.reset_index(drop=False)
Tavg=Tavg[1:]
Tavg
prec=prec.rename(columns={0:1991,1:1992,2:1993,3:1994,4:1995,5:1996,6:1997,7:1998,8:1999,9:2000,10:2001})
prec.reset_index(drop=False)
prec=prec[1:]
prec
prec.describe()
plt.figure(figsize=(10,10))
grid= gsp.GridSpec(4,3)
grid.update(hspace=1)



temp = plt.subplot(grid[0:2,:])
pre = plt.subplot(grid[2:,:])

temp.plot(Tavg[1991],'-',color='blue',linewidth=0.5)
temp.plot(Tavg[1992],'-',color='g',linewidth=0.5)
temp.plot(Tavg[1993],'-',color='r',linewidth=0.5)
temp.plot(Tavg[1994],'-',color='c',linewidth=0.5)
temp.plot(Tavg[1995],'-',color='m',linewidth=0.5)
temp.plot(Tavg[1996],'-',color='grey',linewidth=0.5)
temp.plot(Tavg[1997],'-',color='k',linewidth=0.5)
temp.plot(Tavg[1998],'-',color='beige',linewidth=0.5)
temp.plot(Tavg[1999],'-',color='lightgreen',linewidth=0.5)
temp.plot(Tavg[2000],'-',color='orange',linewidth=0.5)
temp.plot(Tavg[2001],'-',color='pink',linewidth=0.5)
temp.set_xlabel('Months',size=12)
temp.set_ylabel('Temperature (Celsius)',size=12)
temp.spines['top'].set_visible(False)
temp.spines['right'].set_visible(False)
temp.set_xlim('Jan','Dec')
#temp.legend(title='Years',fancybox=True,facecolor='lightgrey',loc=1)
temp.set_title('Monthly mean of Temperatures for each \n year in Kolkata between 1991-2001',weight='bold',fontsize=12,fontfamily='serif')


pre.plot(Tavg[1991],'-',color='blue',linewidth=0.5)
pre.plot(Tavg[1992],'-',color='g',linewidth=0.5)
pre.plot(Tavg[1993],'-',color='r',linewidth=0.5)
pre.plot(Tavg[1994],'-',color='c',linewidth=0.5)
pre.plot(Tavg[1995],'-',color='m',linewidth=0.5)
pre.plot(Tavg[1996],'-',color='grey',linewidth=0.5)
pre.plot(Tavg[1997],'-',color='k',linewidth=0.5)
pre.plot(Tavg[1998],'-',color='beige',linewidth=0.5)
pre.plot(Tavg[1999],'-',color='lightgreen',linewidth=0.5)
pre.plot(Tavg[2000],'-',color='orange',linewidth=0.5)
pre.plot(Tavg[2001],'-',color='pink',linewidth=0.5)
pre.set_xlabel('Months',size=12)
pre.set_ylabel('Precipitation (mm)',size=12)
pre.spines['top'].set_visible(False)
pre.spines['right'].set_visible(False)
pre.set_xlim('Jan','Dec')
#pre.legend(title='Years',fancybox=True,facecolor='lightgrey',loc=1)
pre.set_title('Monthly mean of Precipitation for each \n year in Kolkata between 1991-2001',weight='bold',fontsize=12,fontfamily='serif')

box1 = temp.get_position()
temp.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])
temp.legend(loc='center left', bbox_to_anchor=(1, 0.5),title='Years',fancybox=True,facecolor='lightgrey')

box2 = pre.get_position()
pre.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])
pre.legend(loc='center left', bbox_to_anchor=(1, 0.5),title='Years',fancybox=True,facecolor='lightgrey')


