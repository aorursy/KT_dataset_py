import matplotlib.pyplot as plt

%matplotlib inline
import numpy as np
x = np.arange(0,10)

y = np.arange(11,21)
plt.scatter(x,y,c='g')
plt.plot(x,y,'r*--')
x = np.arange(0,11)

y = x*x

plt.plot(x,y,'b<--')
x = [10,20,14,7,18,12,5]

y = [1,2,3,4,5,6,7]

plt.bar(x,y)

plt.show()
x = np.arange(0,8*np.pi,0.1)

y1 = np.sin(x)

y2 = np.cos(x)

y3 = np.tan(x)

plt.figure(figsize=(20,20))

plt.subplot(3,1,1)

plt.plot(x,y1,color='b')

plt.title('Sine\n',color='green')

plt.subplot(3,1,2)

plt.plot(x,y2,color='g')

plt.title('\nCosine\n',color='green')

plt.subplot(3,1,3)

plt.plot(x,y3,color='r')

plt.title('Tangent\n',color='green')
plt.figure(figsize=(18,9))

labels = 'Python','Java','C++','Ruby','R'

sizes = [215,130,245,120,200]

colors = ['b','r','y','g','purple']

explode = (0.1,0,0,0.1,0)

plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True)

plt.axis('equal')

plt.show()