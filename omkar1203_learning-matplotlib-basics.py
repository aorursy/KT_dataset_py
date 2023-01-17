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
# load libraries

import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])    # only 1 value is given hence it is considered as y value 

plt.ylabel('some numbers')  

plt.show()
plt.plot([1, 2, 3, 4], [1, 4, 9, 16]) # first value is of x and other of y 
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')  # r-red and o-circle

plt.axis([0, 6, 0, 20])

plt.show()

# Here we can use different colors and shapes to mark the points 

""""

colors

r - red              

b - dark blue

g - green

y - yellow

w - white

m - pruple

c - light blue

k - black



shapes

o - circle

s - square

- - lines

-- - dash

^ - traingle

< - left sided traingle

'> - right sided traingle

'* - star 

x - cross

"""
data = {'a': np.arange(50),  # int no from 0 to 49

        'c': np.random.randint(0, 50, 50), # random 50 int no betn 0 and 49 

        'd': np.random.randn(50)} # random 50 float no 

data['b'] = data['a'] + 10 * np.random.randn(50)

data['d'] = np.abs(data['d']) * 100



plt.scatter('a', 'b', c='c', s='d', data=data)    # c - color && s - size 

plt.xlabel('entry a')

plt.ylabel('entry b')

plt.show()
names = ['group_a', 'group_b', 'group_c']

values = [1, 10, 100]



plt.figure(figsize=(9, 3))



plt.subplot(131)

plt.bar(names, values)

plt.subplot(132)

plt.scatter(names, values)

plt.subplot(133)

plt.plot(names, values)

plt.suptitle('Categorical Plotting')

plt.show()
names = ['group_a', 'group_b', 'group_c']

values = [1, 10, 100]

plt.plot(names, values, linewidth=5.0)  # the last output and this one is same but the width of line here is modified 
names = ['group_a', 'group_b', 'group_c']

values = [1, 10, 100]

line, = plt.plot(names, values, '-')

line.set_antialiased(False) # turn off antialiasing
lines = plt.plot(names, values)

# use keyword args

plt.setp(lines, color='r', linewidth=6.0)

# or MATLAB style string value pairs

plt.setp(lines, 'color', 'r', 'linewidth', 6.0)
def f(t):

    return np.exp(-t) * np.cos(2*np.pi*t)



t1 = np.arange(0.0, 5.0, 0.1)

t2 = np.arange(0.0, 5.0, 0.02)



plt.figure()

plt.subplot(211)

plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')



plt.subplot(212)

plt.plot(t2, np.cos(2*np.pi*t2), 'r--')

plt.show()
import matplotlib.pyplot as plt

plt.figure(1)                # the first figure

plt.subplot(211)             # the first subplot in the first figure

plt.plot([1, 2, 3])

plt.subplot(212)             # the second subplot in the first figure

plt.plot([4, 5, 6])





plt.figure(2)                # a second figure

plt.plot([4, 5, 6])          # creates a subplot(111) by default



plt.figure(1)                # figure 1 current; subplot(212) still current

plt.subplot(211)             # make subplot(211) in figure1 current

plt.title('Easy as 1, 2, 3') # subplot 211 title
mu, sigma = 100, 15

x = mu + sigma * np.random.randn(10000)



# the histogram of the data

n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)





plt.xlabel('Smarts')

plt.ylabel('Probability')

plt.title('Histogram of IQ')

plt.text(60, .025, r'$\mu=100,\ \sigma=15$')

plt.axis([40, 160, 0, 0.03])

plt.grid(True)

plt.show()
# Data to plot

labels = 'Python', 'C++', 'Ruby', 'Java'

sizes = [215, 130, 245, 210]

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

explode = (0.4, 0, 0, 0)  # explode 1st slice



# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=False)



plt.axis('equal')

plt.show()
## Bar plot



x = [2,8,10] 

y = [11,16,9]  



x2 = [3,9,11] 

y2 = [6,15,7] 

plt.bar(x, y) 

plt.bar(x2, y2, color = 'g') 

plt.title('Bar graph') 

plt.ylabel('Y axis') 

plt.xlabel('X axis')  



plt.show()
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27]) 

plt.hist(a) 

plt.title("histogram") 

plt.show()
x=np.arange(0,10)

y=np.arange(11,21)



a=np.arange(40,50)

b=np.arange(50,60)



##plotting using matplotlib 



##plt scatter



plt.scatter(x,y,c='g')

plt.xlabel('X axis')

plt.ylabel('Y axis')

plt.title('Graph in 2D')

plt.savefig('Test.png')
y=x*x



## plt plot



plt.plot(x,y,'r*',linestyle='dashed',linewidth=2, markersize=12)

plt.xlabel('X axis')

plt.ylabel('Y axis')

plt.title('2d Diagram')
## Creating Subplots



plt.subplot(2,2,1)

plt.plot(x,y,'r--')

plt.subplot(2,2,2)

plt.plot(x,y,'g*--')

plt.subplot(2,2,3)

plt.plot(x,y,'bo')

plt.subplot(2,2,4)

plt.plot(x,y,'go')