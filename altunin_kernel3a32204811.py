import matplotlib.pyplot as plt

import numpy as np

import math



#1

testArray = [1,2,3,5,6,10,1]

value = np.sum(np.cos(testArray)**2) /2

print(value)





#2

w = x = np.arange(1,10)

b = 10;

value_for_task_2 = np.sum(np.dot(w,x)+b)

print(value_for_task_2)





#3

y = np.arange(0,10)

t = np.arange(10,20)

value_for_task_3 = np.sum((y-t)**2)

print(value_for_task_3)



#5

value_for_task_5 = np.linspace(0.1,10)

plt.plot(value_for_task_5,1/value_for_task_5)

plt.show()



#6

def func(x,N):

    h = float((x-1)/N)

    result = 0.

    x_i = 0.

    for i in range(1,N):

        if x_i<=x:

            x_i = 1+(i*h)

            result+=float(1/x_i)

        else:

            break

    return h*result



resultArray = []

for i in range(1,10):

    resultArray.insert(i,func(i,i))



plt.plot(resultArray)



#11

a=2

k=4

teta = 0

value = np.arange(0,2*math.pi,0.01)

rho = a*np.cos(k*value*teta)

def polar2cart(phi):

    x=rho*np.cos(phi)

    y = rho*np.sin(phi)

    return(x,y)

coord_x,coord_y = polar2cart(value)

figure = plt.figure(figsize=(8,6))

ax=figure.add_subplot(111)

ax.plot(coord_x,coord_y)

plt.grid(True)

plt.show()



plt.subplot(111,polar = True)

phi = np.arange(0,2*math.pi)

rho = a*np.cos(k*phi*teta)

plt.plot(phi,rho,lw =3)

plt.show



#10




