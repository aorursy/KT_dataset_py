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
import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('../input/lb6.csv',sep='\t')

df



df = np.loadtxt('../input/lb6.csv', delimiter=',', skiprows=1)

df
total_profit= df[:,-1] #динамика продаж

fig, ax = plt.subplots(figsize=(20, 7))

ax.set_xlabel('Month Number')

ax.set_ylabel('Total profit')

plt.plot(total_profit);

total_profit= df[:,-1] #динамика продаж

fig, ax = plt.subplots(figsize=(20, 7))

ax.set_xlabel('Month Number')

ax.set_ylabel('Total profit')

ax.plot(total_profit,linestyle='dotted',linewidth=3,color="red",marker='o',markersize=40,label=r"Profit data of last year")

plt.legend(loc=4);
fig, ax = plt.subplots(figsize=(20, 7))

x=df[:,0]

y1=df[:,1]

y2=df[:,2]

y3=df[:,3]

y4=df[:,4]

y5=df[:,5]

y6=df[:,6]

ax.plot(x, y1, linewidth=2, label=r"facecream")

ax.plot(x, y2, linewidth=2, label=r"facewash")

ax.plot(x, y3, linewidth=2, label=r"toothpaste")

ax.plot(x, y4, linewidth=2, label=r"bathingsoap")

ax.plot(x, y5, linewidth=2, label=r"shampoo")

ax.plot(x, y6, linewidth=2, label=r"moisturizer")

ax.set_xlabel('Month number')

ax.set_ylabel('Sales units in data')

plt.legend(loc=0);
fig, ax = plt.subplots(figsize=(20, 7))

plt.scatter(df[:,4],df[:,5] , marker='*', s=100)

plt.grid(True)

ax.set_xlabel('bathing soap')

ax.set_ylabel('shampoo');


x=df[:,0]

y1=df[:,1]#facecream

y2=df[:,2]#facewash

fig, ax = plt.subplots(figsize=(8, 4))

width=0.35

rects1 = ax.bar(x-width/2,y1, width,

                label='Facewash')

rects2 = ax.bar(x+width/2, y2 , width, 

                label='Facecream')



fig.tight_layout()



ax.set_xlabel('Months')

ax.set_title('Sales')

ax.set_xticks(range(1,13))

ax.set_xticklabels([str(i) for i in range(1,13)])

ax.legend()



plt.show();
y1=df[:,1]

y2=df[:,2]

y3=df[:,3]

y4=df[:,4]

y5=df[:,5]

y6=df[:,6]

total_profit= df[:,-1]

totalprofit=np.sum(total_profit)

facecream=np.sum(y1)

facewash=np.sum(y2)

tooth=np.sum(y3)

soap=np.sum(y4)

shampoo=np.sum(y5)

moisturizer=np.sum(y6)



z1=facecream/totalprofit

z2=facewash/totalprofit

z3=tooth/totalprofit

z4=soap/totalprofit

z5=shampoo/totalprofit

z6=moisturizer/totalprofit

z=np.array([z1,z2,z3,z4,z5,z6])*100

fig1, ax1 = plt.subplots(figsize=(20, 7))

labels='facecream','facewash','toothpaste','bathingsoap','shampoo','moisturizer'

ax1.pie(z, labels=labels)

# ax1.pie(z, shadow=True, startangle=90,autopct='%1.1f%%',explode=z,labels=labels)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show();
z=[z1,z2,z3,z4,z5,z6]

z
fig, ax = plt.subplots(2,1, sharex=True,figsize=(10, 7))

x=df[:,0]

y7=df[:,7]

y8=df[:,8]

ax[0].plot(x, y7,  linewidth=2, label=r"(total_units)")

ax[0].legend(loc=0)

ax[1].plot(x, y8,  linewidth=2, label=r"(total_profit)")

ax[1].legend(loc=0)

fig.tight_layout();
