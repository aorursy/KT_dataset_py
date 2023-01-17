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
from matplotlib import pyplot as plt
plt.plot([1,2,3],[4,5,1])
plt.show()

from matplotlib import pyplot as plt
x=[5,8,10]
y=[12,16,6]
plt.plot(x,y)
plt.title('info')
plt.ylabel('y axis')
plt.xlabel('x axis')
plt.show

from matplotlib import pyplot as plt
from matplotlib import style
style.use("ggplot")
x=[5,8,10]
y=[12,16,6]
x2=[6,9,11]
y2=[6,15,7]
plt.plot(x,y,'g',label='line one',linewidth=5)
plt.plot(x2,y2,'g',label='line two',linewidth=5)
plt.title('epic info')
plt.ylabel('y axis')
plt.xlabel('x axis')
plt.legend()
plt.grid(True,color='k')

from matplotlib import pyplot as plt
plt.bar([1,3,5,7,9],[5,2,7,8,2],label="first")
plt.bar([2,4,6,8,10],[8,6,2,5,6],label="second",color='g')
plt.legend()
plt.xlabel('bar number')
plt.ylabel('bar height')
plt.title('my plot')
plt.show()

from matplotlib import pyplot as plt
population=[22,55,62,45,21,22,34,42,42,4,99,102,110,120,121,122,130,111,115,112,80,75,65,54,44,43,42,48]
bins=[0,10,20,30,40,50,60,70,80,90,100,110,120,130]
plt.hist(population,bins,histtype='bar',rwidth=0.8)
plt.xlabel('x')
plt.ylabel('y')
plt.title('histogram')
plt.legend()
plt.show()

from matplotlib import pyplot as plt
x=[1,2,3,4,5,6,7,8]
y=[5,2,4,2,1,4,5,2]
plt.scatter(x,y,label='skitscat',color='k',s=25,)
plt.xlabel('x')
plt.ylabel('y')
plt.title('scatter plot')
plt.legend()

from matplotlib import pyplot as plt
days=[1,2,3,4,5]
sleeping=[7,8,6,11,7]
eating=[2,3,4,3,2]
working=[7,8,7,2,2]
playing=[8,5,7,8,13]
plt.plot([],[],color='m',label=sleeping,linewidth=5)
plt.plot([],[],color='m',label=eating,linewidth=5)
plt.plot([],[],color='m',label=working,linewidth=5)
plt.plot([],[],color='m',label=playing,linewidth=5)
plt.stackplot(days,sleeping,eating,working,playing,colors=['m','c','r','k'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('stck plot')
plt.legend()
plt.show()

from matplotlib import pyplot as plt
slices=[7,2,2,13]
activities=['sleeping','eating','working','playing']
cols=['c','m','r','b']
plt.pie(slices,
labels=activities,
colors=cols,
startangle=90,
shadow=True,
explode=(0,0.1,0,0),
autopct='%1.1f%%')
plt.title('pie plot')
plt.show
import numpy as np
from matplotlib import pyplot as plt
def f(t):
    return np.exp(-t)*np.cos(2*np.pi*t)
t1=np.arange(0.5,5.0,0.1)
t2=np.arange(0.5,5.0,0.2)
plt.subplot(221)
plt.plot(t1,f(t1),'bo',t2,f(t2))
plt.subplot(222)
plt.plot(t2,np.cos(2*np.pi*t2))
plt.show

