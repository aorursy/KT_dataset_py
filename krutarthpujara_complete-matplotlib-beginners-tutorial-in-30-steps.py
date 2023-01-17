#Import Dependencies

import pandas as pd

import numpy as np



#Import Matplotlib

import matplotlib.pyplot as plt
%matplotlib inline
# Example 1 :



#Create Evenly Spaced Numbers between Range: np.linspace(Start, Stop, Intervals)

x1 = np.linspace(0, 10, 100)



# create a plot figure

fig = plt.figure()



#Plot Sin and Cosine of values in x1

plt.plot(x1, np.sin(x1), '-')

plt.plot(x1, np.cos(x1), '--');
# create a plot figure

plt.figure()





# create the first of two panels and set current axis

plt.subplot(2, 1, 1)   # (rows, columns, panel number)

plt.plot(x1, np.sin(x1))





# create the second of two panels and set current axis

plt.subplot(2, 1, 2)   # (rows, columns, panel number)

plt.plot(x1, np.cos(x1));

# get current figure information



print(plt.gcf())
# get current axis information



print(plt.gca())
plt.plot([1, 2, 3, 4])

plt.ylabel('Numbers')

plt.show()
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])

plt.show()
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')

plt.axis([0, 6, 0, 20]) #[xmin, xmax, ymin, ymax]

plt.show()
# evenly sampled time at 200ms intervals

t = np.arange(0., 5., 0.2)



# red dashes, blue squares and green triangles

plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

plt.show()
#Create data to plot

x1 = np.linspace(0, 10, 100)



#Create Figure and Axes

fig, axes1 = plt.subplots(figsize=(8, 4))



#Using Stateless/Object Oriented Methods that are Implicitly called by StateFul Plotly Functions

axes1.plot(x1, np.sin(x1), label=('Sine'))

axes1.plot(x1-1, np.cos(x1), label=('Cosine'));



axes1.set_title('Sine Vs Cosine')

axes1.set_xlabel('Time')

axes1.set_ylabel('Magnitude')

axes1.legend(loc=(0.84, 0.85))

fig.show()
# Plotly functions used with fig object implicitly uses Axes methods

# Thus fig.axes[0] is actually axes1

(fig.axes[0] is axes1)
x = np.linspace(0, 2, 100)



plt.plot(x, x, label='linear')

plt.plot(x, x**2, label='quadratic')

plt.plot(x, x**3, label='cubic')



plt.xlabel('x label')

plt.ylabel('y label')



plt.title("Simple Plot")



plt.legend()



plt.show()
# First create a grid of plots

# ax will be an array of two Axes objects

fig, ax = plt.subplots(2)





# Call plot() method on the appropriate object

ax[0].plot(x1, np.sin(x1), 'b-')

ax[1].plot(x1, np.cos(x1), 'b-');

fig = plt.figure()



x2 = np.linspace(0, 5, 10)

y2 = x2 ** 2



axes = fig.add_axes([0.1, 0.1, 0.8, 0.9])



axes.plot(x2, y2, 'r')



axes.set_xlabel('x2')

axes.set_ylabel('y2')

axes.set_title('title');
fig = plt.figure()



ax = plt.axes()
plt.plot([1, 3, 2, 4])#, 'b-')

plt.show()





# We Gave One Set of Values, It took the values as Y-axis values

#For X-axis values it took default values=[0,1,2,3] Corresponding to Y-axis having 4 values
x3 = np.arange(0.0, 6.0, 0.01) 

plt.plot(x3, [xi**2 for xi in x3], 'b-') 



plt.show()
x4 = range(1, 5)



plt.plot(x4, [xi*1.5 for xi in x4])



plt.plot(x4, [xi*3 for xi in x4])



plt.plot(x4, [xi/3.0 for xi in x4])



# Add as many lines you want before calling Show()

plt.show()
# Saving the figure

fig.savefig('plot1.png')
# Explore the contents of figure



from IPython.display import Image

Image('plot1.png')
# Explore supported file formats





fig.canvas.get_supported_filetypes() 
# Create figure and axes first

fig = plt.figure()



ax = plt.axes()



# Declare a variable x5

x5 = np.linspace(0, 10, 1000)





# Plot the sinusoid function

ax.plot(x5, np.sin(x5), 'b-'); 
# Create figure and axes first

fig = plt.figure()



ax = plt.axes()



# Declare a variable x5

x5 = np.linspace(0, 10, 50)





# Plot the sinusoid function

ax.plot(x5, np.sin(x5), 'o' , color='orange'); 
data1 = np.random.randn(1000)



plt.hist(data1); 
data2 = [5. , 25. , 50. , 20.]

plt.bar(range(len(data2)), data2)

plt.show() 
data2 = [5. , 25. , 50. , 20.]

plt.barh(range(len(data2)), data2)

plt.show() 
x9 = np.arange(0, 4, 0.2)

y9 = np.exp(-x9)

e1 = 0.1 * np.abs(np.random.randn(len(y9)))

plt.errorbar(x9, y9, yerr = e1, fmt = '.-')

plt.show();
A = [15., 30., 45., 22.]

B = [15., 25., 50., 20.]



z2 = range(4)



plt.bar(z2, A, color = 'b')

plt.bar(z2, B, color = 'r', bottom = A)



plt.show()
plt.figure(figsize=(7,7))



x10 = [35, 25, 20, 20]



labels = ['Computer', 'Electronics', 'Mechanical', 'Chemical']



plt.pie(x10, labels=labels);



plt.show()
data3 = np.random.randn(100)



plt.boxplot(data3)



plt.show()
# Create a matrix

matrix1 = np.random.rand(10, 20)



cp = plt.contour(matrix1)



plt.show()
plt.contour(np.random.rand(5,5))
# View list of all available styles



print(plt.style.available)
# Set styles for plots

plt.style.use('seaborn-bright')
x15 = np.arange(1, 5)

plt.plot(x15, x15*1.5, x15, x15*3.0, x15, x15/3.0)



plt.grid(True)



plt.show()
x15 = np.arange(1, 5)

plt.plot(x15, x15*1.5, x15, x15*3.0, x15, x15/3.0)



print(plt.axis())   # shows the current axis limits values

plt.axis([1, 4, 0, 12]) #Xmin, Xmax, Ymin, Ymax ; Check this on graph

print(plt.axis())   # Shows the current axis limits values 



plt.show()
x15 = np.arange(1, 5)



plt.plot(x15, x15*1.5, x15, x15*3.0, x15, x15/3.0)



plt.xlim([0, 5.0])

plt.ylim([0.0, 15.0])
u = [5, 4, 9, 7, 8, 9, 6, 5, 7, 8]

plt.plot(u)



plt.xticks([2, 4, 6, 8, 10])

plt.yticks([2, 4, 6, 8, 10])



plt.show()
plt.plot([1, 3, 2, 4])



plt.xlabel('Axes Label the X axis')

plt.ylabel('Axes Label the Y axis')



plt.show()
plt.plot([1, 3, 2, 4])

plt.title('First Plot')



# plt.show() #Commenting this shows us the location of the Title
x15 = np.arange(1, 5)



fig, ax = plt.subplots()

ax.plot(x15, x15*1.5)

ax.plot(x15, x15*3.0)

ax.plot(x15, x15/3.0)



ax.legend(['Normal','Fast','Slow']);
x15 = np.arange(1, 5)



fig, ax = plt.subplots()



ax.plot(x15, x15*1.5, label='Normal')

ax.plot(x15, x15*3.0, label='Fast')

ax.plot(x15, x15/3.0, label='Slow')



# ax.legend(loc=0); #Method 1

ax.legend(loc=(0.35,0.77)) #Method 2
x16 = np.arange(1, 5)



plt.plot(x16, 'r')

plt.plot(x16+1, 'g')

plt.plot(x16+2, 'b')



plt.show()
x16 = np.arange(1, 5)



plt.plot(x16, '--', x16+1, '-.', x16+2, ':')



plt.show()