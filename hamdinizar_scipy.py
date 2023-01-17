import numpy as np 

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
x = np.linspace ( 0, 10, 10)

plt.scatter(x,x**2)



f = interp1d(x,x**2, kind ='linear')



new_x = np.linspace(0,10,30)

result=f(new_x)



plt.scatter(new_x,result,c='red')
x = np.linspace(0,10,10)

y = np.sin(x)



f = interp1d(x,y, kind ='cubic')



new_x = np.linspace(0,10,30)

result=f(new_x)

plt.scatter(x,y)

plt.scatter(new_x,result,c='red')
x = np.linspace(0,2,100)

y = 1/3*x**3 - 3/5*x**2 + 2 + np.random.randn(x.shape[0])/20

plt.scatter(x,y)

plt.xlabel('x')

plt.ylabel('y')
def f(x,a,b,c,d):

    return a*x**3 + b*x**2 + c*x + d    # model
from scipy import optimize
param , param_cov = optimize.curve_fit(f,x,y)

plt.scatter(x,y)

plt.plot(x,f(x,param[0],param[1],param[2],param[3]),c='g',lw = 3)
def f(x):

    return x**2 + 15*np.sin(x)
x = np.linspace(-10,10,100)

plt.plot(x,f(x))
optimize.minimize(f,x0=-3) 
x0=-3

result = optimize.minimize(f,x0=x0).x 

plt.plot(x,f(x),lw=3,zorder=-1)

plt.scatter(result,f(result),s=100,c='g',zorder=1)

plt.scatter(x0,f(x0),marker='+',s=200,c='r',zorder=1)

plt.show()

def f (x):

    return np.sin(x[0]) + np.cos(x[0]+x[1])*np.cos(x[0])
x=np.linspace(-3,3,100)

y=np.linspace(-3,3,100)



x,y = np.meshgrid(x,y)

plt.contour(x,y,f(np.array([x,y])),20)



x0 = np.zeros((2,1))

plt.scatter(x0[0],x0[1], marker ='+', c = 'r', s = 100)





result = optimize.minimize(f,x0=x0).x 



plt.scatter (result[0],result[1],c='g',s=100)



print(result)
x = np.linspace(0,30,1000)

y = 3*np.sin(x) + 2*np.sin(5*x) + np.sin(10*x) + np.random.random(x.shape[0])*10

plt.plot(x,y)
from scipy import fftpack
fourier = fftpack.fft(y)



power = np.abs(fourier)



freq = fftpack.fftfreq(y.size)



plt.plot(np.abs(freq),power)
fourier [power<400] = 0 
plt.plot(np.abs(freq),np.abs(fourier))
filtred_signal = fftpack.ifft(fourier)
plt.figure(figsize=(12,8))

plt.plot(x,y,lw=0.5,label='original signal')

plt.plot(x,filtred_signal,lw=3,label='filtred signal')

plt.legend()

plt.show()
from scipy import ndimage 
np.random.seed(0)

x = np.zeros((32,32))

x[10:-10,10:-10]=1

x[np.random.randint(0,32,30),np.random.randint(0,32,30)] = 1

plt.imshow(x)
open_x=ndimage.binary_opening(x)

plt.imshow(open_x)