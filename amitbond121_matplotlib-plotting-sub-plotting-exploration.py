import matplotlib.pyplot as plt

%pylab inline 
import numpy as np

x=np.random.rand(100)

print (x[:10])

plt.plot(x)
'''We will work with two arrays this time and plot.linsspace afunction from numpy.

Here were generating venly sequences number from -pi to +pi,length of array is 256.

Array y is sin of each value of x'''

x=np.linspace(-np.pi,np.pi,256,endpoint=True)

y=np.sin(x)

plt.plot(x,y)
'''Multi line plot'''

x=np.linspace(-np.pi,np.pi,256,endpoint=True)

y1=np.sin(x)

y2=np.cos(x)



plt.plot(x,y1)

plt.plot(x,y2)

'''Plotting a standard normal distribution.Scatter plot is important thing for exploratory data Analysis'''

x=np.random.randn(100)

#y=a+bx+cx2+error y linear function of x

y=10+12*x+4*x**2+np.random.normal(0,10,100)



print ("x: ")

print (x[:10])



print("y: ")

print (y[:10])



plt.scatter(x,y)
'''Creating empty figure object.Subplotting using matplotlib'''

x=np.random.randn(1000)

plt.figure(figsize=(6,6),dpi=120)



plt.subplot(221)# make a 2x2 grid and subplot at1 

plt.plot(x,color='red',linewidth=0.3,linestyle='-')



plt.subplot(222)# make a 2x2 grid and subplot at2

y=10+12*x+np.random.normal(0,5,1000)

plt.scatter(x,y,color='green')

            

plt.subplot(223)

plt.plot(y)

            

plt.subplot(224)

plt.hist(y,color='cyan')




