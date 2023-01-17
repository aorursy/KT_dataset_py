import numpy as np

import matplotlib.pyplot as plt
# help(np.random.randint)

# age=

np.random.randint(20,30,size=100)
# height=

np.random.normal(1.5,0.12,size=100)

# list(map(lambda x:'%.2f' % x,np.random.normal(1.5,0.12,size=100)))
np.array([ [1,2],[3,4] ])

# np.array([age,height])
np.random.randint(101,size = (20,3))
x=np.random.randint(-10,11,size=[10,10])

x
x[x<0]=0

x
row=range(len(x))

col=range(len(x))

x[row,col]=0

x
x =np.array([1,2,3])

x+1

y=np.array([[1],[2],[3]])

x+y
a = np.array([[1,-1],[1,1]])

b = np.array([30,66])

sol = np.linalg.solve(a, b)

sol
np.e
def euler_e(n):

    return (1+1/n)**n

display(euler_e(1),euler_e(10),euler_e(1000),np.e)
n=100

print("Total value with n =",n,"\b:",np.sum(np.ones(n)/np.arange(1,n+1)**2))



print("Actual value with n is infinity:",np.pi**2/6)
np.euler_gamma
num=100; x=np.arange(num)

err= 5*np.random.normal(size=[1,num]).flatten()

y = x+err



# Compute the area using the composite trapezoidal rule.

area = np.trapz(y, dx=1)

print("Approximate area using trapezoidal rule:", area)

print("Area of a triangle:",0.5*100*100)
# Look at other functions in NumPy

print(dir(np))