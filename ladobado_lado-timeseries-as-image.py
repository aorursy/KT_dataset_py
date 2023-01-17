import matplotlib.pyplot as plt
import numpy as np
import math

data1,data2,data3,data4,data5,data6 = [],[],[],[],[],[]

for i in range(1,100):
    data1.append(np.ones(2)*i)
    data2.append(np.ones(2)*(i+10))
    data3.append(np.ones(2)*(i*i))
    data4.append(np.ones(2)*((i+10)*(i+10)))
    alfa = i*2*math.pi/100
    #print(np.ones(10)*(np.sin(3*alfa)+np.sin(7*alfa))/np.sin(alfa))
    data5.append(np.ones(2)*((np.sin(3*alfa)+np.sin(7*alfa))/np.sin(alfa)))
    data6.append(np.ones(2)*(np.log(i)))
                 
#print(data1)
import sklearn as sk
import sklearn.metrics.pairwise

def recurrence_plot(data, eps=None, steps=None):
    if eps==None: eps=.1
    if steps==None: steps=10000
    #takes each row (coordinates) from data (data must be two dimensional array) and measures euclidian distance with all other rows (coordinates) inclusive the same row
    '''
    data = [[0,0],[1,1]]
    sk.metrics.pairwise.pairwise_distances(data)
    
    [[0.         1.41421356]
     [1.41421356 0.        ]]
    '''
    
    d = sk.metrics.pairwise.pairwise_distances(data)
    d = np.floor(d / eps)
    #d[d > steps] = steps

    return d

fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(1,6,1)
ax.set(title='x')
ax.imshow(recurrence_plot(data1))

ax = fig.add_subplot(1,6,2)
ax.set(title='x+100')
ax.imshow(recurrence_plot(data2))

ax = fig.add_subplot(1,6,3)
ax.set(title='x*x')
ax.imshow(recurrence_plot(data3))

ax = fig.add_subplot(1,6,4)
ax.set(title='(x+100)*(x+100)')
ax.imshow(recurrence_plot(data4))

ax = fig.add_subplot(1,6,5)
ax.set(title='(sin(3*x)+sin(7*x))/sin(x)')
ax.imshow(recurrence_plot(data5))

ax = fig.add_subplot(1,6,6)
ax.set(title='(log(x)')
ax.imshow(recurrence_plot(data6))

def GAF_plot(data):
    """Compute the Gramian Angular Field of an image"""
    # Min-Max scaling in interval <-1,1>
    min_ = np.amin(data)
    max_ = np.amax(data)
    scaled_data = ((data - max_)+(data - min_))/(max_ - min_)
    
    # Polar encoding
    phi = np.arccos(scaled_data)

    # GAF Computation (every term of the matrix)
    xx, yy = np.meshgrid(phi, phi, sparse=True)    

    return np.cos(xx+yy)

    
fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(1,6,1)
ax.set(title='x')
ax.imshow(GAF_plot(data1))

ax = fig.add_subplot(1,6,2)
ax.set(title='x+100')
ax.imshow(GAF_plot(data2))

ax = fig.add_subplot(1,6,3)
ax.set(title='x*x')
ax.imshow(GAF_plot(data3))

ax = fig.add_subplot(1,6,4)
ax.set(title='(x+100)*(x+100)')
ax.imshow(GAF_plot(data4))

ax = fig.add_subplot(1,6,5)
ax.set(title='(sin(3*x)+sin(7*x))/sin(x)')
ax.imshow(GAF_plot(data5))

ax = fig.add_subplot(1,6,6)
ax.set(title='(log(x)')
ax.imshow(GAF_plot(data6))