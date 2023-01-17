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
print('Hello world')
import matplotlib.pyplot as plt
x= np.array([x for x in range(0,10)])
y= np.array([x for x in range(0,10)])
plt.scatter(x,y,c='r')
plt.show()
plt.scatter(x,y,c='r')
plt.xlabel('X-axis')
plt.ylabel('y-axis')
plt.title('Graph')

plt.show()
plt.savefig("2dGraph.png")
y=x*x
plt.plot(x,y,'r*',linestyle='solid',linewidth=2,markersize=12)
plt.xlabel('X-axis')
plt.ylabel('y-axis')
plt.title('Graph')

plt.show()
plt.subplot(2,2,1)
plt.plot(x,y ,'r*')
plt.subplot(2,2,2)
plt.plot(x,y ,'g--')
plt.subplot(2,2,3)
plt.plot(x,y ,'bo')
plt.subplot(2,2,4)
plt.plot(x,y ,'go')
plt.show()
y=5*x+5
plt.title("2d Graph")
plt.plot(x,y)
plt.show()
x= np.arange(0,4*np.pi,0.01)
y= 2*0.4*np.sin(x)
plt.plot(x,y,'r--')
plt.show()
#Subplot()
# Compute the x and y coordinates for points on sine and cosine curves 
x = np.arange(0, 5 * np.pi, 0.1) 
y_sin = np.sin(x) 
y_cos = np.cos(x)  
   
# Set up a subplot grid that has height 2 and width 1, 
# and set the first such subplot as active. 
plt.subplot(2, 1, 1)
   
# Make the first plot 
plt.plot(x, y_sin,'r--') 
plt.title('Sine')  
   
# Set the second subplot as active, and make the second plot. 
plt.subplot(2, 1, 2) 
plt.plot(x, y_cos,'g--') 
plt.title('Cosine')  
   
# Show the figure. 
plt.show()
#subplot()
# compute the x and y coordinates for points on sine and cosine curves
x=np.arange(0,5*np.pi,0.9)
y_sin=np.sin(x)
y_cos=np.cos(x)
# Set up a subplot grid that has height 2 and width 1, 
# and set the first such subplot as active.
plt.subplot(2, 1, 1)
# Make the first plot 
plt.plot(x, y_sin,'r--') 
# Set the second subplot as active, and make the second plot. 
plt.subplot(2, 1, 2) 
plt.bar(x, y_cos,color='g') 
plt.title('Cosine')  
   
# Show the figure. 
plt.show()
x = [2,8,10] 
y = [11,16,9]  

x2 = [3,9,10] 
y2 = [6,15,7] 
plt.bar(x, y,color='r') 
plt.bar(x2, y2, color = 'g')
plt.show()
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27]) 
plt.hist(a) 
plt.title("histogram") 
plt.show()
x=np.arange(0,5*np.pi,0.9)
y_sin=np.sin(x)
plt.hist(y_sin,color='r')
plt.show()
data = [np.random.normal(0, std, 100) for std in range(1, 4)]

# rectangular box plot
plt.boxplot(data,vert=True,patch_artist=False);
labels = ['python','java','c++','javascript']
sizes = [100,80,90,100]
exclude = (0.4,0,0,0)
colors = ['gold', 'y', 'r', 'b']
plt.pie(sizes, explode=exclude, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True)

plt.axis('equal')
plt.show()
