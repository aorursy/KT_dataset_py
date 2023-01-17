%matplotlib inline



import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

from shapely.geometry import LineString

from statistics import mean



from sympy import *

#from sympy import symbols

#from sympy import plot



from math import sqrt


position_data = {'t (s)': [0, 1, 2, 3, 4, 5], 'x(m)': [30, 52, 38, 0,-37,-53]}

position_DataFrame= pd.DataFrame.from_dict(position_data)

position_DataFrame



# We can print, min, max, mean, average of postion

print(f"The max value of the position is: {position_DataFrame['x(m)'].max()}")

print(f"The min value of the position is: {position_DataFrame['x(m)'].min()}")

print(f"The mean value of the position is: {position_DataFrame['x(m)'].mean()}")

print(f"The median value of the position is: {position_DataFrame['x(m)'].median()}")



position_DataFrame.plot(kind='scatter',x='t (s)', y='x(m)');


# calculates speed from postion and time

position_DataFrame['v(m/s)'] = (position_DataFrame['x(m)'] - position_DataFrame['x(m)'].shift(1)) / (position_DataFrame['t (s)'] - position_DataFrame['t (s)'].shift(1))

position_DataFrame



# Replace null values by zero

position_DataFrame =position_DataFrame.fillna(0)

position_DataFrame



#Calculate acceleration from speed

position_DataFrame['a(m/s^2)'] = (position_DataFrame['v(m/s)'] - position_DataFrame['v(m/s)'].shift(1)) / (position_DataFrame['t (s)'] - position_DataFrame['t (s)'].shift(1))

position_DataFrame =position_DataFrame.fillna(0)

position_DataFrame



# form data frame out of time points

time = pd.DataFrame({'t':range(0,6)})

time['y'] = 0 + 20 * time['t'] -  0.5* 9.8*time['t'] **2 

time.plot(kind='scatter',x='t',y='y');

#print time to see the data frame



# line plot

time = pd.DataFrame({'t':range(0,6)})

time['y'] = 0 + 20 * time['t'] - 0.5* 9.8*time['t'] **2  

time.plot(kind='line',x='t',y='y');

# using \delta t = T/N

time =10

dt = 0.1;

n = int(round(time/dt))

t = np.linspace(0,5,n) # time axis with n-points



y = 0+20*t-0.5*9.8*t**2 #here is our equation



# Create the plot

plt.plot(t, y,label='Position y in meters', color ='m')



# Add a title

plt.title('Position versus Time')





# Add X and y Label

plt.xlabel('t in sec')

plt.ylabel('y in meters')



# Add a grid

plt.grid(alpha=.4,linestyle='--')



# Add a Legend

plt.legend()



# Show the plot

plt.show()
x=0





time =10

dt = 0.1;

n = int(round(time/dt))

t = np.linspace(0,4,n)

# Create the vectors X and Y

#t = np.array(range(0,5,n))

# choses very few data so better make it smooth with ff 

x = x-4*t+2*t**2



# Create the plot

plt.plot(t, x,label='Position versus Time', color='c')







# Add a title

plt.title('Position versus Time')



# Add limit

#plt.xlim()

plt.ylim(-2.1,10.1)



# Add X and y Label

plt.xlabel('t in sec')

plt.ylabel('x meters')



# Add a grid

plt.grid(alpha=.4,linestyle='--')



# Add a Legend

plt.legend()



# Show the plot

plt.show()


t = Symbol('t')

x = -4*t+2*t**2

v=x.diff(t) # dx/dt

a = v.diff(t)# dv/dt

print(f"The positon is: {x}")

print(f"The speed  is: {v}")

print(f"The acceleration  is: {a}")





p = plot(x, v, a, (t, 0, 5.6), show=false)

#change the color of p's

p[0].line_color = 'b' # x

p[1].line_color = 'r' # v

p[2].line_color = 'g' # a



p.show()



# Use cmath instead of math if you are working with complex numbers

#import cmath



a = 4

b = -42

c = -2000

d = (b**2) - (4*a*c)

root1 = (-b-sqrt(d))/(2*a)

root2 = (-b+sqrt(d))/(2*a)

print(root1)

print(root2)



# informations we gathered from the question 

xA=0

xB=2000

vA=0

vB=-42

aA=5.6

aB=2.4



# calculate distance traveled by Car A and Car B respectively

time=10

dt = 0.1;

n = int(round(time/dt))

t = np.linspace(0,40,n)

xA = xA  + vA*t+0.5*aA *t**2

xB = xB + vB*t+0.5*aB *t**2

# Create the plot

plt.plot(t, xA,label='CarA', color='r')

plt.plot(t, xB,label='CarB', color ='b')





# Add a title

plt.title('Distance of CarA and CarB')



# Add X and y Label

plt.xlabel('time in seconds')

plt.ylabel('distance in meters')





# Add a grid

plt.grid(alpha=.4,linestyle='--')



# Add a Legend

plt.legend()



# Show the plot

plt.show()


# informations we gathered from the question 

xA=0

xB=2000

vA=0

vB=-42

aA=5.6

aB=2.4





time=10

dt = 0.1;

n = int(round(time/dt))

t = np.linspace(0,40,n)

xA = xA  + vA*t+0.5*aA *t**2

xB = xB + vB*t+0.5*aB *(t)**2

# Create the plot

plt.plot(t, xA,label='CarA', color='r')

plt.plot(t,xB,label='CarB', color='b')





# Add a title

plt.title('Distance of CarA and CarB')



# Add X and y Label

plt.xlabel('time in seconds')

plt.ylabel('distance in meters')



# Add a grid

plt.grid(alpha=.4,linestyle='--')



# Add a Legend

plt.legend()



# Show the plot



#To show intersection

first_line = LineString(np.column_stack((t, xA)))

second_line = LineString(np.column_stack((t, xB)))

intersection = first_line.intersection(second_line)



if intersection.geom_type == 'MultiPoint':

    plt.plot(*LineString(intersection).xy, 'o', color = 'black')

elif intersection.geom_type == 'Point':

    plt.plot(*intersection.xy, 'o', color = 'black')

plt.show()
xA, xB = intersection.xy # use this if it is single point

#xA, xB = LineString(intersection).xy

print(xA,xB)
a1 = 1

b1 = -32

c1 = 1

d1 = (b1**2) - (4*a1*c1)

rootC = (-b1-sqrt(d1))/(2*a1)

rootT = (-b1+sqrt(d1))/(2*a1)

print(rootC)

print(rootT)
#Given

xC=45

xT=0

vC=45

vT=0

aC=0

aT=3





dt = 0.1;#0.00001

n = int(round(time/dt))

t = np.linspace(0,40,n)

xC = xC  + vC*t+0.5*aC *t**2

xT = xT + vT*t+0.5*aT *t**2

# Create the plot

plt.plot(t, xC,label='Car', color='b')

plt.plot(t,xT,label='Tropper', color='r')





# Add a title

plt.title('Distance of Car and Tropper')



# Add X and y Label

plt.xlabel('t in s')

plt.ylabel('distance in meters')



# Add limit

#plt.xlim(-1,32)

#plt.ylim(0,1600)



# Add a grid

plt.grid(alpha=.4,linestyle='--')



# Add a Legend

plt.legend()



# Show the plot



#To show intersection

first_line = LineString(np.column_stack((t, xC)))

second_line = LineString(np.column_stack((t, xT)))

intersection = first_line.intersection(second_line)



if intersection.geom_type == 'MultiPoint':

    plt.plot(*LineString(intersection).xy, 'o', color = 'black')

elif intersection.geom_type == 'Point':

    plt.plot(*intersection.xy, 'o', color = 'black')

plt.show()
xC, xT = intersection.xy # use this if it is single point

#x, y = LineString(intersection).xy

print(xC,xT)