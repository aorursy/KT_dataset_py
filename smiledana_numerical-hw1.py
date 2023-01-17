# Import some other libraries that need

import matplotlib

import matplotlib

import numpy as np

import matplotlib.pyplot as plt



# Design variables at mesh points

x = np.arange(-5.0, 10.0, 0.02)

y = np.arange(-5.0, 10.0, 0.02)

x1, x2 = np.meshgrid(x, y)



# Equations and Constraints

z = x2-8/x1

y1 = 0.2*x1-x2

y2 = 16-(x1-5)**2-x2**2



# Create a contour plot

plt.figure()

# Weight contours

lines = np.linspace(0.8,3.0,5.0)

CS = plt.contourf(x1,x2,z,lines)

artists, labels = CS.legend_elements()

plt.legend(artists, labels, handleheight=2,loc=2)





# y1

CS1 = plt.contour(x1,x2,y1,[0.0],colors='r',linestyles='dashed',extend='max',origin='upper')

CS1 = plt.contourf(x1,x2,y1,[0.0,10.0],colors='none',linestyles='dashed',extend='max',origin='upper',

                   hatches=['\\\\'])

# y2

CS2 = plt.contour(x1,x2,y2,[0.0],colors='b',linestyles='dashed',extend='max',origin='upper')

CS2 = plt.contourf(x1,x2,y2,[0.0,10.0],colors='none',linestyles='dashed',extend='max',origin='upper',

                   hatches=['/'])

#Import some other libraries that need

import matplotlib

import matplotlib

import numpy as np

import matplotlib.pyplot as plt



# Design variables at mesh points

x = np.arange(-5.0, 10.0, 0.02)

y = np.arange(-5.0, 10.0, 0.02)

x1, x2 = np.meshgrid(x, y)



# Equations and Constraints

z = x2-8/x1

y1 = 0.2*x1-x2

y2 = 16-(x1-5)**2-x2**2



# y1

CS1 = plt.contour(x1,x2,y1,[0.0],colors='r',linestyles='dashed',extend='max',origin='upper')

CS1 = plt.contourf(x1,x2,y1,[0.0,10.0],colors='none',linestyles='dashed',extend='max',origin='upper',

                   hatches=['\\\\'])

# y2

CS2 = plt.contour(x1,x2,y2,[0.0],colors='b',linestyles='dashed',extend='max',origin='upper')

CS2 = plt.contourf(x1,x2,y2,[0.0,10.0],colors='none',linestyles='dashed',extend='max',origin='upper',

                   hatches=['/'])

#1.6(a)

# Import some other libraries that need

# matplotlib and numpy packages must also be installed

import matplotlib

import matplotlib

import numpy as np

import matplotlib.pyplot as plt



# Design variables at mesh points

x = np.arange(-10.0, 40.0, 0.02)

y = np.arange(-10.0, 40.0, 0.02)

x1, x2 = np.meshgrid(x, y)



# Equations and Constraints

z = 30-x2+x1

y1 = -2*x1-3*x2+30

y2 = x1

y3 = x2



# Create a contour plot

plt.figure()

# Weight contours

lines = np.linspace(0.0,20.0,5.0)

CS = plt.contourf(x1,x2,z,lines)

artists, labels = CS.legend_elements()

plt.legend(artists, labels, handleheight=0.2,loc=7)

# y1

CS1 = plt.contourf(x1,x2,y1,[0.0,1.0],colors='r')

# y2

CS2 = plt.contour(x1,x2,y2,[0.0],colors='b',linestyles='dashed')

CS2 = plt.contourf(x1,x2,y2,[0.0,10.0],colors='none',linestyles='dashed',extend='max',origin='upper',

                   hatches=['\\\\'])

# y3

CS3 = plt.contour(x1,x2,y3,[0.0],colors='g',linestyles='dashed')

CS3 = plt.contourf(x1,x2,y3,[0.0,10.0],colors='none',extend='max',origin='upper',hatches=['/'])

#1.6(b)

# Import some other libraries that need

# matplotlib and numpy packages must also be installed

import matplotlib

import matplotlib

import numpy as np

import matplotlib.pyplot as plt



# Design variables at mesh points

x = np.arange(-10.0, 40.0, 0.02)

y = np.arange(-10.0, 40.0, 0.02)

x1, x2 = np.meshgrid(x, y)



# Equations and Constraints

z = 30-x2+x1

y1 = -2*x1-3*x2+30

y2 = x1

y3 = x2



# y1

CS1 = plt.contourf(x1,x2,y1,[0.0,1.0],colors='r')

# y2

CS2 = plt.contour(x1,x2,y2,[0.0],colors='b',linestyles='dashed')

CS2 = plt.contourf(x1,x2,y2,[0.0,10.0],colors='none',linestyles='dashed',extend='max',origin='upper',

                   hatches=['\\\\'])

# y3

CS3 = plt.contour(x1,x2,y3,[0.0],colors='g',linestyles='dashed')

CS3 = plt.contourf(x1,x2,y3,[0.0,10.0],colors='none',extend='max',origin='upper',hatches=['/'])


