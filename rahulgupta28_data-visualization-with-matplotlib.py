# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import the matplotlib.pyplot module 

import matplotlib.pyplot as plt

%matplotlib inline
import numpy as np

x= np.linspace(0,90, 5000)

y = (x ** 4)/2
print(x)

print("*" * 50)

print(y)
plt.plot(x,y,'*')

plt.xlabel('X data')

plt.ylabel('Y data')

plt.title('Ploating data X vs Y')
plt.subplot(1,5,1)

plt.plot(x,y,'r-')

plt.subplots_adjust(right=2)

plt.subplot(1,5,2)

plt.plot(y,x,'g-')

plt.subplot(1,5,3)

plt.plot(x**5,y**2,'g-')

plt.subplot(1,5,4)

plt.plot(y**6,x,'b-')

plt.subplot(1,5,5)

plt.plot(y*x,x,'g-')
#Matplotlib having object oriented api , we can instantiate figure object and then call methods or attributes from that object



fig = plt.figure()

axes = fig.add_axes([1,1,1,1])

axes.plot(x,y*2)

axes.set_xlabel('x axes')

axes.set_ylabel('y axes')
fig = plt.figure()

axes1 = fig.add_axes([.1,.1,.8,.8])

axes2 = fig.add_axes([.3,.5,.3,.3])

axes1.plot(x,y**2,'g')

axes2.plot(y,x**2,'b')

axes1.set_xlabel('x axes')

axes1.set_ylabel('y axes')

axes2.set_xlabel('y axes')

axes2.set_ylabel('x axes')

axes1.set_title('x vs y - square')

axes2.set_title('y vs x - square ')



fig = plt.figure()



ax = fig.add_axes([0,0,1,1])



ax.plot(x, x**2, label="x**2")

ax.plot(x, x**3, label="x**3")

ax.legend()
fig = plt.figure()



ax = fig.add_axes([0,0,1,1])



ax.plot(x, x**2, label="x**2")

ax.plot(x, x**3, label="x**3")

ax.legend(loc=1)  #ax.legend(loc=1) # upper right corner , ax.legend(loc=2) # upper left corner  ,ax.legend(loc=3) # lower left corner ,ax.legend(loc=4) # lower right corner
x = np.linspace(0, 5, 11)

y = x ** 2

plt.scatter(x,y)
from random import sample

data = sample(range(1, 2000), 30)

plt.hist(data)
data = [np.random.normal(0, std, 100) for std in range(1, 4)]

#  box plot

plt.boxplot(data,vert=True,patch_artist=True);   