#Create a figure of size 8 inches in width, and 6 inches in height. Name it as fig.

#Create an axis, associated with figure fig, using add_subplot. Name it as ax.

#Create a list t with values [5, 10, 15, 20, 25].

#Create a list d with values [25, 50, 75, 100, 125].

#Draw a line, by plotting t values on X-Axis and d values on Y-Axis. Use plot function. Label the line as d = 5t

#Label X-Axis as time (seconds)

#Label Y-Axis as distance (meters)

#Set Title as Time vs Distance Covered

#Limit data points on X-Axis from 0 to 30.

#Limit data points on Y-Axis from 0 to 130.

#Add a legend
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(111)

t = [5, 10, 15, 20, 25]

d = [25, 50, 75, 100, 125]

ax.set(title='Time vs Distance Covered',

      xlabel='time (seconds)', ylabel='distance (meters)',

      xlim=(0, 30), ylim=(0,130))

plt.plot(t, d, label='d = 5t')

plt.legend()

plt.show()
#1 test_sine_wave_plot



#Create a figure of size 12 inches in width, and 3 inches in height. Name it as fig.

#Create an axis, associated with figure fig, using add_subplot. Name it as ax.

#Create a numpy array t having 200 values between 0.0 and 2.0 . Use 'linspace' method to generate 200 values.

#Create a numpy array v, such that v = np.sin(2.5*np.pi*t).

#Pass t and v as variables to plot function and draw a red line passing through choosen 200 points. Label the line as sin(t).

#Label X-Axis as Time (seconds)

#Label Y-Axis as Voltage (mV)

#Set Title as Sine Wave

#Limit data on X-Axis from 0 to 2.

#Limit data on Y-Axis from -1 to 1.

#Mark major ticks on X-Axis at 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, and 2.0

#Mark major ticks on Y-Axis at -1, 0, and 1.

#Add a grid, whose linestyle is '--'.

#Add a legend
import numpy as np
import matplotlib.ticker as ticker

fig = plt.figure(figsize=(12,3))

ax = fig.add_subplot(111)

t = np.linspace(0.0,2.0,200)

v = np.sin(2.5*np.pi*t)

ax.set(title='Sine Wave',

      xlabel='time (seconds)', ylabel='Voltage (mV)',

      xlim=(0, 2), ylim=(-1,1))

xmajor = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8,2.0]

ymajor = [-1,0,1]

ax.xaxis.set_major_formatter(ticker.FixedFormatter(xmajor))

ax.yaxis.set_major_formatter(ticker.FixedFormatter(ymajor))

plt.plot(t, v, label='sin(t)',color = 'red')

plt.legend()

plt.show()
#2 test_multi_curve_plot



#Create a figure of size 12 inches in width, and 3 inches in height. Name it as fig.

#Create an axis, associated with figure fig, using add_subplot. Name it as ax.

#Create a numpy array x having 20 values between 0.0 and 5.0 . Use 'linspace' method to generate 20 values.

#Create three numpy arrays y1, y2 and y3 using the expressions : y1 = x, y2 = x**2 and y3 = x**3 respectively.

#Draw a red colored line passing through x and y1, using plot function. Mark the 20 data points on the line as circles.Label the line as y = x.

#Draw a green colored line passing through x and y2, using plot function. Mark the 20 data points on the line as squares. Label the line as y = x**2.

#Draw a blue colored line passing through x and y3, using plot function. Mark the 20 data points on the line as upward pointed triangles.

#Label the line as y = x**3.

#Label X-Axis as X

#Label Y-Axis as f(X)

#Set Title as Linear, Quadratic, & Cubic Equations.

#Add a Legend.
fig = plt.figure(figsize=(12,3))

ax = fig.add_subplot(111)

x = np.linspace(0.0,5.0,20)

y1 = x

y2 = x**2

y3 = x**3



ax.set(title='Linear, Quadratic, & Cubic Equations',

      xlabel='X', ylabel='f(x)')

ax.plot(x, y1, label='y=x',marker = 'o',color = 'red')

ax.plot(x, y2, label='y = x**2',marker = "s",color = 'green')

ax.plot(x, y3, label='y = x**3',marker = "^",color = 'blue')

plt.legend()

plt.show()
#3 test_scatter_plot



#Create a figure of size 12 inches in width, and 3 inches in height. Name it as fig.

#Create an axis, associated with figure fig, using add_subplot. Name it as ax.

#Consider the list s = [50, 60, 55, 50, 70, 65, 75, 65, 80, 90, 93, 95]. It represent the number of cars sold by a Company 'X' in each month of 2017, starting from Jan, 2017.

#Create a list, months, having numbers from 1 to 12.

#Draw a scatter plot with variables months and s as it's arguments. Mark the data points in red color. Use scatter function for plotting.

#Limit data on X-Axis from 0 to 13.

#Limit data on Y-Axis from 20 to 100.

#Mark ticks on X-Axis at 1, 3, 5, 7, 9, and 11.

#Label the X-Axis ticks as Jan, Mar, May, Jul, Sep, and Nov respectively.

#Label X-Axis as Months

#Label Y-Axis as No. of Cars Sold

#Set Title as "Cars Sold by Company 'X' in 2017".

#Follow the steps mentioned in next step to execute the code.
fig = plt.figure(figsize=(12,3))

ax = fig.add_subplot(111)

s = [50, 60, 55, 50, 70, 65, 75, 65, 80, 90, 93, 95]

months = [1,2,3,4,5,6,7,8,9,10,11,12]

ax.set(title="Cars Sold by Company 'X' in 2017",

      xlabel='Months', ylabel='No. of Cars Sold',xlim=(0, 13), ylim=(20,100))

ax.scatter(months, s,marker = 'o',color = 'red',edgecolors = 'black')

plt.xticks([1, 3, 5, 7, 9,11])

ax.set_xticklabels(['Jan', 'Mar', 'May', 'Jul', 'Sep','Nov'])

plt.show()