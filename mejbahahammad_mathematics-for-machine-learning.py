x = -41

x + 16 == -25
x = 4

3*x - 2 == 10
x = 7

5*x + 1 - 2*x == 22
x = 45

x/3 + 1 == 16
x = 25

2/5 * x + 1 ==11
x = 1.5

3*x + 2 == 5*x -1
x = 2

4*(x + 2) + 3*(x - 2) == 16
import pandas as pd



# Create a dataframe with an x column containing values from -10 to 10

df = pd.DataFrame ({'x': range(-10, 11)})



# Add a y column by applying the solved equation to x

df['y'] = (3*df['x'] - 4) / 2



#Display the dataframe

df.head()
%matplotlib inline

from matplotlib import pyplot as plt



plt.plot(df.x, df.y, color="grey", marker = "o")

plt.xlabel('x')

plt.ylabel('y')

plt.grid()

plt.show()
plt.plot(df.x, df.y, color="grey")

plt.xlabel('x')

plt.ylabel('y')

plt.grid()



## add axis lines for 0,0

plt.axhline()

plt.axvline()

plt.show()
plt.plot(df.x, df.y, color="grey")

plt.xlabel('x')

plt.ylabel('y')

plt.grid()



## add axis lines for 0,0

plt.axhline()

plt.axvline()

plt.annotate('x-intercept',(1.333, 0))

plt.annotate('y-intercept',(0,-2))

plt.show()
plt.plot(df.x, df.y, color="grey")

plt.xlabel('x')

plt.ylabel('y')

plt.grid()

plt.axhline()

plt.axvline()



# set the slope

m = 1.5



# get the y-intercept

yInt = -2



# plot the slope from the y-intercept for 1x

mx = [0, 1]

my = [yInt, yInt + m]

plt.plot(mx,my, color='red', lw=5)



plt.show()
%matplotlib inline



import pandas as pd

from matplotlib import pyplot as plt



# Create a dataframe with an x column containing values from -10 to 10

df = pd.DataFrame ({'x': range(-10, 11)})



# Define slope and y-intercept

m = 1.5

yInt = -2



# Add a y column by applying the slope-intercept equation to x

df['y'] = m*df['x'] + yInt



# Plot the line

from matplotlib import pyplot as plt



plt.plot(df.x, df.y, color="grey")

plt.xlabel('x')

plt.ylabel('y')

plt.grid()

plt.axhline()

plt.axvline()



# label the y-intercept

plt.annotate('y-intercept',(0,yInt))



# plot the slope from the y-intercept for 1x

mx = [0, 1]

my = [yInt, yInt + m]

plt.plot(mx,my, color='red', lw=5)



plt.show()