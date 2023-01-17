import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, exp, pi
from scipy.integrate import quad
def f(x):
    #return x
    #return 0.5*x*(x**2 + x - 2)
    #return 3*(x**3)-(8*(x**2))-(3*x)+12
    return 8 - 2*x
print(f(-1))
def g(x):
    #return 7/2 * x**2 + x - 3
    #return 4*np.sin((np.pi/12) * x * (x - 1))
    #return x*(x + 2)
    #return 4*e**((x+1)*(x-3))+2*x+2
    return x**3 - 7*x**2 + 12*x
t = np.arange(1, 4.1, 0.1)
fig, ax = plt.subplots()
ax.plot(t, f(t))
ax.plot(t, g(t))
ax.grid(True, linestyle='-')
ax.tick_params(labelcolor='r', labelsize='medium', width=3)
plt.show()
vals = [[f(x), g(x)] for x in np.arange(3.3, 3.4, 0.01)]
print(vals)
# call quad to integrate f from 1 to 2
#res1, err1 = quad(f, 0, 2)
#res1, err1 = quad(f, -2, 0)
#res1, err1 = quad(f, -1, 1)
res1, err1 = quad(g, 1, 2)
#print("The numerical result is {:f} (+-{:g})".format(res1, err1))

# call quad to integrate h from -2 to 0
#res2, err2 = quad(g, 0, 2)
#res2, err2 = quad(g, -2, 0)
#res2, err2 = quad(g, -1, 1)
res2, err2 = quad(f, 1, 2)
#print("The numerical result is {:f} (+-{:g})".format(res2, err2))
firstPart = res1 - res2
print("The first enclosed area is {:f}".format(firstPart))
# call quad to integrate g from 2 to 3.39464
#call quad to integrate from 2 to 4
# 0 to 3
#res3, err3 = quad(g, 2, 3.39464)
#res3, err3 = quad(g, 0, 3)
res3, err3 = quad(g, 1, 3)
res3, err3 = quad(f, 2, 4)
#print("The numerical result is {:f} (+-{:g})".format(res3, err3))

# call quad to integrate f from 2 to 3.39464
# 0 to 3
#res4, err4 = quad(f, 2, 3.39464)
#res4, err4 = quad(f, 0, 3)
#res4, err4 = quad(f, 1, 3)
res4, err4 = quad(g, 2, 4)
#print("The numerical result is {:f} (+-{:g})".format(res4, err4))
secondPart = res3 - res4
print("The second enclosed area is {:f}".format(secondPart))
#print("The area between the two lines from x = 0 to 3.39464 is {:f}".format(firstPart + secondPart))
#print("The area between the two lines from x = -2 to 3 is {:f}".format(firstPart + secondPart))
#print("The area between the two lines from x = -1 to 3 is {:f}".format(firstPart + secondPart))
print("The area between the two lines from x = 1 to 4 is {:f}".format(firstPart + secondPart))
from pylab import *
x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
def c(x):
    return np.cos(x)

def s(x):
    return np.sin(x)

res5, err5 = quad(c, -3*np.pi/4, np.pi/4)
res6, err6 = quad(s, -3*np.pi/4, np.pi/4)
print("The area between the two lines from x = -3pi/4 to pi/4 is {:f}".format(res5-res6))
c,s = np.cos(x), np.sin(x)
plot (x,c), plot(x,s)
show()
print("Twice the square root of 2 is : {:f}".format(np.sqrt(2)*2))