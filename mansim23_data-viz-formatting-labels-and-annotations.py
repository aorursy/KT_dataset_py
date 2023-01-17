# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Data Viz: Formatting Plots
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import seaborn as sb
import matplotlib.pyplot as plt
from pylab import rcParams
#parameters
%matplotlib inline
rcParams['figure.figsize']=5,4
sb.set_style('whitegrid')
#objects
x=range(1,10)
y=[1,2,3,4,0.5,4,3,2,1]
plt.bar(x,y)
wide=[0.5,0.5,0.5,0.9,0.9,0.9,0.5,0.5,0.5,]
color= ["red"]
plt.bar(x,y,width=wide, color=color, align='center')
address='../input/mtcars.csv'
cars=pd.read_csv(address)
df=cars[['cyl','mpg','wt']]
df.plot()
#changing the colors above
color_theme=['darkgray','powderblue','lightsalmon']
df.plot(color=color_theme)
#making a pie chart
z=[1,2,3,4,0.5]
plt.pie(z)
color_theme=['#A9A9A9','#FFA07A','yellow','blue','green']
plt.pie(z,colors=color_theme)
#Customizing line styles
x1=range(0,10)
y1=[10,9,8,7,6,5,4,3,2,1]
plt.plot(x,y)
plt.plot(x1,y1)

#ls=line style, lw=width
plt.plot(x,y,ls='steps',lw=5)
plt.plot(x1,y1,ls='--',lw=10)
#Adding markers to plots. mew=marker width, marker is the kind of marker
plt.plot(x,y,marker='1',mew=10)
plt.plot(x1,y1,marker='*',mew=10)
#Labels and Annotations for a. Functional and b.Object oriented method using the .annotate(xy,xytext,arrowprop) where xy
#is the location being annotated and arrowprop is the dictionary of arrow properties

#.legend(label and location)
x2=range(1,10)
y2=[1,2,3,4,0.5,4,3,2,1]
plt.bar(x2,y2)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
#Labels for a pie chart eg. labels of vehicles
z=[1,2,3,4,0.5]
veh_type=['bike','car','cycle','truck','scooter']
plt.pie(z,labels=veh_type)
#Object Oriented plotting
address='../input/mtcars.csv'
cars=pd.read_csv(address)
cars.columns=['car_names','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']
#Create a series called mpg
mpg=cars.mpg
fig=plt.figure()
ax=fig.add_axes([.1,.1,1,1])
mpg.plot()
ax.set_xticks(range(32))
ax.set_xticklabels(cars.car_names,rotation=90, fontsize='medium')
ax.set_title('Cars in mtCars')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
#Adding a legend to a plot using the .legend and location can be best, upper left, lower right, etc
plt.pie(z)
plt.legend(veh_type, loc="lower right")
address='../input/mtcars.csv'
cars=pd.read_csv(address)
cars.columns=['car_names','mpg','cyl','disp','hp','drat','wt','qsec','vs','am','gear','carb']
#Create a series called mpg
mpg=cars.mpg
fig=plt.figure()
ax=fig.add_axes([.1,.1,1,1])
mpg.plot()
ax.legend(loc='best')
#Annotations
mpg.max()
fig=plt.figure()
ax=fig.add_axes([0.1,0.1,1,1])
mpg.plot()
ax.set_title('Mileage')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_ylim([0,45])
ax.annotate('Toyota Corolla', xy=(19,33.9), xytext=(21,35),arrowprops=dict(facecolor='black', shrink=0.05))
