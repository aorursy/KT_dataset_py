# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
petrol=pd.read_csv('/kaggle/input/curude-oil-diesel-petrol-and-lpg-prices/Indian petrolium product prices - Diesel.csv')
petrol.head()
data=pd.read_csv('/kaggle/input/marksheetoffirstyear/IIT Marksheet.csv')
data.head()
import pandas as pd
import matplotlib.pyplot as plt

labels=list(data['Student ID'])
thermo=list(data['Thermodynamics'])
Lang=list(data['Language'])

width=0.5

fig=plt.figure(figsize=(20,10))
ax=plt.subplot()

ax.bar(labels,thermo,width,yerr=0,label='Thermodynamics')
ax.bar(labels,Lang,width,yerr=0,bottom=thermo,label='Language')

ax.set_ylabel('marks out of 100')
ax.set_title('comparer plotter\n')
plt.xticks(rotation=45)
ax.legend()

plt.show()



t=np.arange(0,30,0.01)
print(t)
nse=np.random.randn(len(t))
print(nse)
pi=np.pi
print(pi)
#plotting the coherent of two signals
np.random.seed(19680801)

step=0.01
t=np.arange(0,30,step)           #creates evenly spaced array

noise1=np.random.randn(len(t))   #generate random noise 
noise2=np.random.randn(len(t))

pi=np.pi
signal1=np.sin(2*pi*10*t)+noise1 #generate signal with random noise
signal2=np.sin(2*pi*10*t)+noise2

fig, ng=plt.subplots(2,1,figsize=(10,7))

ng[0].plot(t,signal1,t,signal2)
ng[0].set_xlim(0,2)
ng[0].set_xlabel('time')
ng[0].set_ylabel('signals:1/2')
ng[0].grid(True)

cxy, f=ng[1].cohere(signal1,signal2, 256, 1./step) 
ng[1].set_ylabel('coherency')

fig.tight_layout()                                 

plt.show()
#stem plot
x=np.linspace(0,2*np.pi,40)
y=np.exp(np.sin(x))

plt.figure(figsize=(10,7))
plt.stem(x,y,use_line_collection=True)
plt.show


#stackplot
x=list(data['Student ID'])
y1=list(data['Thermodynamics'])
y2=list(data['Language'])
y3=list(data['EVS'])

y=np.vstack([y1,y2,y3])
labels=['Thermodynamics','Language','EVS']

fig, ng=plt.subplots(figsize=(10,7))
ng.stackplot(x,y,labels=labels)
plt.xticks(rotation=90)
ng.legend(loc='upper left')

plt.show
x=list(petrol['Date'].head())
y=list(petrol['Kolkata'].head())
z=list(petrol['Mumbai'].head())

plt.figure(figsize=(20,5))
plt.step(x, y , label='delhi')
plt.plot(x, y , 'o--', color='grey', alpha=0.3)

plt.step(x,z, label='mumbai')
plt.plot(x,z,'o--', color='grey',alpha=0.3)

plt.grid(axis='x' , color='0.75')
plt.title('petrol hike plotter')
plt.xticks(rotation=90)
plt.legend(title='city')
plt.show()