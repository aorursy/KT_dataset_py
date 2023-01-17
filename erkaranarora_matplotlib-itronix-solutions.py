import matplotlib.pyplot as plt



X = [5,10,15,20,25,30,35]

Y = [14,64,24,78,45,98,50]

Z = [98,67,45,34,23,56,38]



plt.xlim(0,40)

plt.ylim(0,120)



plt.text(18,113,"Itronix",fontsize=15,fontname='Times New Roman',color='#F000db',weight='bold')



plt.title("Suicides Dataset India",fontsize=15,fontname='Times New Roman',color='#05f4db',weight='bold')

plt.xlabel("Year",fontsize=15,fontname='Times New Roman',color='black',weight='bold')

plt.ylabel("Total Suicides",fontsize=15,fontname='Times New Roman',color='k',weight='bold')



plt.plot(X,Y,label="Haryana",color='r',linewidth=3,linestyle='dashed')

plt.plot(X,Z,label="Punjab",linewidth=3,marker='^',markerfacecolor='k',markersize=8)



plt.legend()

plt.show()
import matplotlib.pyplot as plt



X = [5,10,15,20,25,30,35]

Y = [14,64,24,78,45,98,50]

Z = [98,67,45,34,23,56,38]



plt.axis([0,40,0,120])



plt.text(18,113,"Itronix",fontsize=15,fontname='Times New Roman',color='#F000db',weight='bold')



plt.title("Suicides Dataset India",fontsize=15,fontname='Times New Roman',color='#05f4db',weight='bold')

plt.xlabel("Year",fontsize=15,fontname='Times New Roman',color='black',weight='bold')

plt.ylabel("Total Suicides",fontsize=15,fontname='Times New Roman',color='k',weight='bold')



plt.plot(X,Y,label="Haryana",color='r',linewidth=3,linestyle='dashed')

plt.plot(X,Z,label="Punjab",linewidth=3,marker='^',markerfacecolor='k',markersize=8)



plt.legend()

plt.savefig('itronix.pdf')

plt.savefig('itronix.png')

plt.show()
import numpy as np

import pandas as pd
x = np.arange(1,16)

x
y = 7 * x + 3

y
plt.plot(x,y,'ro')

plt.plot(x,y**2,'b^')

plt.show()
x = [1,2,3,4,5,6]

y = [10,20,15,18,7,19]



xLabels = ['Jan','Feb','Mar','Apr','May','Jun']



plt.plot(x,y)

plt.xticks(x,xLabels,rotation=45)

plt.show()
import matplotlib.pyplot as plt



X = [5,10,15,20,25,30,35]

Y = [14,64,24,78,45,98,50]

Z = [98,67,45,34,23,56,38]



plt.axis([0,40,0,120])



plt.title('Title',fontsize=15,fontname='Times New Roman',color='k',weight='bold')

plt.xlabel('Time',fontsize=15,fontname='Times New Roman',color='k',weight='bold')

plt.ylabel('Sensor',fontsize=15,fontname='Times New Roman',color='k',weight='bold')



plt.bar(X,Y,width=2.5,label='Punjab',color='r')

plt.bar(X,Z,width=2.5,label='Haryana',color='k',alpha=0.6)



plt.legend()

plt.show()
import matplotlib.pyplot as plt



X = [5,10,15,20,25,30,35]

Y = [14,64,24,78,45,98,50]

Day = ['M','T','W','T','F','S','S']



plt.title('Title',fontsize=15,fontname='Times New Roman',color='k',weight='bold')

plt.xlabel('Time',fontsize=15,fontname='Times New Roman',color='k',weight='bold')

plt.ylabel('Sensor',fontsize=15,fontname='Times New Roman',color='k',weight='bold')



plt.bar(X,Y,width=2.5,label='Punjab',color='r',tick_label=Day)



plt.legend()

plt.show()
M = [20,35,30,35,27]

F = [25,32,34,20,25]

x = np.arange(len(M))



plt.bar(x,M,width=0.35,label='Male')

plt.bar(x+0.35,F,width=0.35,label='Female')



plt.xticks(x+0.35/2,['G1','G2','G3','G4','G5'])



plt.legend()

plt.show()
M = [20,35,30,35,27]

F = [25,32,34,20,25]

x = np.arange(len(M))



plt.title("Title")

plt.xlabel('Gender')

plt.ylabel('Age')



plt.ylim(0,40)



a = plt.bar(x,M,width=0.35,label='Male')

b = plt.bar(x+0.35,F,width=0.35,label='Female')



for value in a:

    height = value.get_height()

    plt.text(value.get_x() + value.get_width()/2,1.04 * height,'%.0f'%float(height),ha='center')

    

for value in b:

    height = value.get_height()

    plt.text(value.get_x() + value.get_width()/2,1.04 * height,'%.0f'%float(height),ha='center')



plt.xticks(x+0.35/2,['G1','G2','G3','G4','G5'])



plt.legend()

plt.show()
import matplotlib.pyplot as plt



X = [5,10,15,20,25,30,35]

Y = [14,64,24,78,45,98,50]

Z = [98,67,45,34,23,56,38]



plt.axis([0,40,0,120])

plt.title('Title',fontsize=15,fontname='Times New Roman',color='k',weight='bold')

plt.xlabel('Time',fontsize=15,fontname='Times New Roman',color='k',weight='bold')

plt.ylabel('Sensor',fontsize=15,fontname='Times New Roman',color='k',weight='bold')



plt.scatter(X,Y,label='Sensor 1')

plt.scatter(X,Z,label='Sensor 2')



plt.legend()

plt.show()
import matplotlib.pyplot as plt



slices = [40,10,10,10,30]

cols = ['red','green','b','cyan','y']

channels = ['9x','Sony','Star Plus','Colors','Cartoon Network']



plt.title('Pie Plot')



plt.pie(slices,colors=cols,labels=channels,explode=(0,0,0.2,0,0),autopct='%.2f%%')



plt.legend(fontsize=12,bbox_to_anchor=(1.65,0.8))

plt.show()
days = [1,2,3,4,5]

sleeping = [7,8,6,11,7]

eating = [2,3,4,3,2]

working = [7,8,7,2,2]

playing = [8,5,7,8,13]



Lab=['Sleeping','Eating','Working','Playing']



plt.title('Week Plan')

plt.xlabel('Days')

plt.ylabel('24 Hours')



plt.stackplot(days,sleeping,eating,working,playing,labels=Lab)

plt.legend(fontsize=15,bbox_to_anchor=(1.4,1.0))

plt.show()
import matplotlib.pyplot as plt

import numpy as np

x = np.arange(10)

y = np.random.randint(10,100,10)

y
plt.title('Plot 1')

plt.xlabel('X Axis')

plt.ylabel('Y Axis')

plt.plot(x,y)

plt.show()
plt.title('Plot 1')

plt.xlabel('X Axis')

plt.ylabel('Y Axis')

plt.plot(x,y**2)

plt.show()
plt.title('Plot 1')

plt.xlabel('X Axis')

plt.ylabel('Y Axis')

plt.plot(x,y**3)

plt.show()
plt.title('Plot 1')

plt.xlabel('X Axis')

plt.ylabel('Y Axis')

plt.plot(x,y**4)

plt.show()
plt.rc('figure',figsize=(18,10))



plt.subplot(2,2,1)

plt.title('Plot 1')

plt.xlabel('X Axis')

plt.ylabel('Y Axis')

plt.plot(x,y,color='k')



plt.subplot(2,2,2)

plt.title('Plot 2')

plt.xlabel('X Axis')

plt.ylabel('Y Axis')

plt.plot(x,y**2,color='r')



plt.subplot(2,2,3)

plt.title('Plot 3')

plt.xlabel('X Axis')

plt.ylabel('Y Axis')

plt.plot(x,y,color='y')



plt.subplot(2,2,4)

plt.title('Plot 4')

plt.xlabel('X Axis')

plt.ylabel('Y Axis')

plt.plot(x,y,color='b')



plt.subplots_adjust(top=0.92,bottom=0.08,left=0.10,right=0.90,hspace=0.25,wspace=0.15)

plt.show()
import matplotlib.pyplot as plt

from matplotlib import gridspec
fig = plt.figure()

gs = gridspec.GridSpec(3,3)



plt1 = fig.add_subplot(gs[0,:])

plt1.plot(x,y,'r--')

plt1.set_title('Plot 1')

plt1.set_xlabel('X axis')

plt1.set_ylabel('Y axis')



plt2 = fig.add_subplot(gs[1,:-1])

plt2.plot(x,y,'b')

plt2.set_title('Plot 2')

plt2.set_xlabel('X axis')

plt2.set_ylabel('Y axis')



plt3 = fig.add_subplot(gs[1:,2])

plt3.plot(x,y,'k')

plt3.set_title('Plot 3')

plt3.set_xlabel('X axis')

plt3.set_ylabel('Y axis')



plt3 = fig.add_subplot(gs[2,0])

plt3.plot(x,y,'k')

plt3.set_title('Plot 3')

plt3.set_xlabel('X axis')

plt3.set_ylabel('Y axis')





plt3 = fig.add_subplot(gs[2,1])

plt3.plot(x,y,'k')

plt3.set_title('Plot 3')

plt3.set_xlabel('X axis')

plt3.set_ylabel('Y axis')
import quandl

data = quandl.get("BSE/BOM539678", authtoken="x8VdDXzBty3iGsaRLTVs")

data
plt.rc('figure',figsize=(15,5))

plt.plot(data.Open)

plt.show()