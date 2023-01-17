from numpy import sin,pi,linspace
from pylab import plot,show,subplot
import matplotlib.pyplot as plt

w2 = [1,1,1,2,3,3,4,5] # plotting the curves for
w1 = [1,2,3,3,4,5,5,6] # different values of a/b
deltas=[0, pi/4, pi/2, 3*pi/4,pi]
t = linspace(0,2*pi,300)
plt.figure(figsize=(12,25))
for i in range(len(w1)):
    for k, delta in enumerate(deltas):
        x = sin(w1[i] * t)
        y = sin(w2[i] * t + delta)
        plt.subplot(8,5,i*len(deltas)+k+1)
        plt.plot(x,y)
plt.show()