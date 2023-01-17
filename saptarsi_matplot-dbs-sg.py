import matplotlib.pyplot as plt

#plt.plot([1,2,3,4])

plt.plot([1,2,3,4],[1,4,9,16],'ro')

plt.title('My first plot')

plt.plot()

plt.plot([1,2,4,2,1,0,1,2,1,4],linewidth=2.0)
# Horizontal Subplots

import numpy as np

t = np.arange(0,5,0.1)

y1 = np.sin(2*np.pi*t)

y2 = np.sin(2*np.pi*t)

plt.subplot(211)

plt.plot(t,y1,'b-.')

plt.subplot(212)

plt.plot(t,y2,'r--')

# Verical Slubplots 121 and 122
plt.axis([0,5,0,20])

plt.title('My first plot')

# plt.title('My first plot',fontsize=20,fontname='Times New Roman')

plt.xlabel('Counting')

plt.ylabel('Square values')

plt.plot([1,2,3,4],[1,4,9,16],'ro')

plt.text(1,1.5,'First')

plt.text(2,4.5,'Second')

plt.legend(['First series'])

# The position of the legend can be changed based on the loc paramtere

# Some values of loc are 
import matplotlib.pyplot as plt

import numpy as np

x = np.arange(-2*np.pi,2*np.pi,0.01)

y = np.sin(3*x)/x

y2 = np.sin(2*x)/x

y3 = np.sin(4*x)/x

plt.plot(x,y)

plt.plot(x,y2)

plt.plot(x,y3)
# Histogram

pop = np.random.randint(0,100,100)

n,bins,patches = plt.hist(pop,bins=20)



# Bar Chart

index = np.arange(5)

values1 = [5,7,3,4,6]

plt.bar(index,values1)

plt.xticks(index+0.4,['A','B','C','D','E'])
# Scatter plot

rng = np.random.RandomState(0)

x = rng.randn(100)

y = rng.randn(100)

colors = rng.rand(100)

sizes = 500 * rng.rand(100)



plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,

            cmap='viridis')

plt.colorbar();  # show color scale
# Import dataset

import pandas as pd

titan=pd.read_csv('../input/train.csv')



# Assignment 1 Draw the Age distribution of titanic passengers Histogram

pop =titan['Age']

plt.title('Titanic Passenger Age')

n,bins,patches = plt.hist(pop,bins=5)





# Assignment 2 Draw the survived and not survived age Histogram

ns_age= titan['Age'][titan['Survived']==0].mean()

s_age = titan['Age'][titan['Survived']==1].mean()

index = np.arange(2)

values1 = [ns_age,s_age]

plt.bar(index,values1)

plt.xticks(index+0.4,['NS','S'])


