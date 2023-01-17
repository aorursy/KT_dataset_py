import matplotlib.pyplot as plt



plt.plot([1,2,3,4])

plt.show()
import matplotlib.pyplot as plt

import numpy as np



x = np.arange(0,5,0.01)

# print(x)



plt.plot(x,[i**2 for i in x])

plt.show()
import matplotlib.pyplot as plt



x = range(5)



plt.plot(x,[i*i for i in x], x, [i*i*i for i in x], x, [i**4 for i in x])

plt.show()
import matplotlib.pyplot as plt



x = range(5)



plt.plot(x,[i*i for i in x], x, [i*i*i for i in x], x, [i**4 for i in x])

plt.grid(True)

plt.show()
from matplotlib import pyplot as plt



x = [5,8,10]

y = [12,16,6]



plt.plot(x,y)



plt.title('Epic Info')

plt.ylabel('Y axis')

plt.xlabel('X axis')



plt.show()
import matplotlib.pyplot as plt

from matplotlib import style



style.use('ggplot')



x = [5,8,10]

y = [12,16,6]



x2 = [6,9,11]

y2 = [6,15,7]



# can plot specifically, after just showing the defaults:

plt.plot(x,y,linewidth=4)

plt.plot(x2,y2,linewidth=4)





# Give Plot Title

plt.title('Epic Info')



# Define both axis level

plt.ylabel("Y axis")

plt.xlabel("X axis")



plt.show()

from matplotlib import pyplot as plt

from matplotlib import style



style.use('ggplot')



x = [5,8,10]

y = [12,16,6]



x2 = [6,9,11]

y2 = [6,15,7]





plt.bar(x, y, align='center')



plt.bar(x2, y2, color='g', align='center')





plt.title('Epic Info')

plt.ylabel('Y axis')

plt.xlabel('X axis')



plt.show()
from matplotlib import pyplot as plt

from matplotlib import style



style.use('ggplot')



x = [5,8,10]

y = [12,16,6]



x2 = [6,9,11]

y2 = [6,15,7]



plt.scatter(x, y)#, align='center')



plt.scatter(x2, y2, color='g')#, align='center')





plt.title('Epic Info')

plt.ylabel('Y axis')

plt.xlabel('X axis')



plt.show()
from matplotlib import pyplot as plt

from matplotlib import style



style.use('ggplot')



x = [5,8,10]

y = [12,16,6]



x2 = [6,9,11]

y2 = [6,15,7]



plt.scatter(x, y, label="Student")#, align='center', label="Stud")



plt.scatter(x2, y2, color='g', label="Marks")#, align='center', label="Marks")



plt.legend()



# Give Plot Title 

plt.title('Epic Info')



# Give title to the both axis

plt.ylabel('Y axis')

plt.xlabel('X axis')



#



plt.show()
import numpy, matplotlib.pyplot as plt



x = numpy.arange(5)



plt.plot(x, x, label="liner", linewidth=2)

plt.plot(x, x*x, label="square", linewidth=2)

plt.plot(x, x*x*x, label="cube", linewidth=2)



plt.grid(True)



plt.ylabel("Y-axis")

plt.xlabel("X-axis")



plt.title("Polynomial Graph")



plt.legend()



plt.show()
import matplotlib.pyplot as plt

import numpy



y = np.random.randn(40,2)



plt.hist(y)

plt.show()
import numpy as np

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

 

x = [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]

num_bins = 5

n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)

plt.show()
import matplotlib.pyplot as plt



plt.bar([2,3.5,5],[2,5,7])

plt.show()
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt

 

objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')

y_pos = np.arange(len(objects))

performance = [10,8,6,4,2,1]

 

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Usage')

plt.title('Programming language usage')

 

plt.show()
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt

 

objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')

y_pos = np.arange(len(objects))

performance = [10,8,6,4,2,1]

 

plt.barh(y_pos, performance, align='center', alpha=0.5)

plt.yticks(y_pos, objects)

plt.xlabel('Usage')

plt.title('Programming language usage')

 

plt.show()
import numpy as np

import matplotlib.pyplot as plt

 

# data to plot

n_groups = 4

means_frank = (90, 55, 40, 65)

means_guido = (85, 62, 54, 20)

 

# create plot

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.35

opacity = 0.8

 

rects1 = plt.bar(index, means_frank, bar_width,

alpha=opacity,

color='b',

label='Frank')

 

rects2 = plt.bar(index + bar_width, means_guido, bar_width,

alpha=opacity,

color='g',

label='Guido')

 

plt.xlabel('Person')

plt.ylabel('Scores')

plt.title('Scores by person')

plt.xticks(index + bar_width, ('A', 'B', 'C', 'D'))

plt.legend()

 

plt.tight_layout()

plt.show()
import matplotlib.pyplot as plt



dictionary = {'A':25, 'B':70, 'C':55, 'D':90}

for i, key in enumerate(dictionary):

  plt.bar(i, dictionary[key])

  

plt.show()
import matplotlib.pyplot as plt

import numpy



dictionary = {'A':25, 'B':70, 'C':55, 'D':90}

for i, key in enumerate(dictionary):

  plt.bar(i, dictionary[key])



plt.xticks(numpy.arange(len(dictionary)),

          dictionary.keys())

  

plt.show()
import matplotlib.pyplot as plt



plt.figure(figsize=(3,3))



x = [40, 20, 5]

labels = ["Bikes", "Cars", "Buses"]



plt.pie(x, labels=labels)

plt.show()
import matplotlib.pyplot as plt

 

# Data to plot

labels = 'Python', 'C++', 'Ruby', 'Java'

sizes = [215, 130, 245, 210]

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

explode = (0.1, 0, 0, 0)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)

 

plt.axis('equal')

plt.show()
import matplotlib.pyplot as plt

 

labels = ['Cookies', 'Jellybean', 'Milkshake', 'Cheesecake']

sizes = [38.4, 40.6, 20.7, 10.3]

colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']

patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)

plt.legend(patches, labels, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()
import matplotlib.pyplot as plt

import numpy as np



x = np.random.rand(1000)

y = np.random.rand(1000)



plt.scatter(x,y)

plt.show()
import numpy as np

import matplotlib.pyplot as plt

 

# Create data

N = 500

x = np.random.rand(N)

y = np.random.rand(N)

colors = (0,0,0)

area = np.pi*3

 

# Plot

plt.scatter(x, y, s=area, c=colors, alpha=0.5)

plt.title('Scatter plot pythonspot.com')

plt.xlabel('x')

plt.ylabel('y')

plt.show()
import matplotlib.pyplot as plt

import numpy as np



y = np.arange(1,3)

plt.plot(y,'y')

plt.plot(y**2, 'm')

plt.plot(y**3, 'c')



plt.show()
import matplotlib.pyplot as plt

import numpy as np



y = np.arange(1,3)

plt.plot(y,'--')

plt.plot(y**2, '-.')

plt.plot(y**3, ':')



plt.show()
import matplotlib.pyplot as plt

import numpy as np



y = np.arange(1,3,0.2)



plt.plot(y, '*', y+0.5, 'o', y+1, 'D', y+2, '^', y+3, 's')

plt.show()
import matplotlib.pyplot as plt

import numpy as np



y = np.arange(1,3,0.2)



plt.plot(y, '*', y*2, 'o', y*3, 'D', y*4, '^', y*5, 's')

plt.show()