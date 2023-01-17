# importing the pyplot

import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])

plt.ylabel('some numbers')

plt.show()

# You may be wondering why the x-axis ranges from 0-3 and the y-axis from 1-4. 

#If you provide a single list or array to the plot() command, matplotlib assumes it is a sequence of y values, and automatically generates the x values for you. Since python ranges start with 0, the default x vector has the same length as y but starts with 0. Hence the x data are [0,1,2,3].

# plot() is a versatile command, and will take an arbitrary number of arguments. For example, to plot x versus y, you can issue the command:

plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
# For every x, y pair of arguments, there is an optional third argument which is the format string that indicates the color and line type of the plot. The letters and symbols of the format string are from MATLAB, and you concatenate a color string with a line style string. 

# The default format string is 'b-', which is a solid blue line. 

# For example, to plot the above with red circles, you would issue.

plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')

plt.axis([0, 6, 0, 20])

plt.show()

# The axis() command takes a list of [xmin, xmax, ymin, ymax] and specifies the viewport of the axes.

# Other format string are 'r--' for Red Dashes; 'bs' for Blue Square; 'g^' is for Green Traingle.
import numpy as np

#x = np.array( [1, 2, 3, 4] )

#y = np.array( [1, 4, 9, 16] )



# Creating an array of range from 0 to 5 with a difference of 0.2

x = np.arange(0., 5., 0.2)



#print (type(x))

plt.plot(x, x, 'r--', x, x**2, 'bs', x, x**3, 'g^')

plt.show()
data = {'a': np.arange(50),

        'c': np.random.randint(0, 50, 50),

        'd': np.random.randn(50)}

data['b'] = data['a'] + 10 * np.random.randn(50)

data['d'] = np.abs(data['d']) * 100



print("Data -->", data)



plt.scatter('a', 'b', c='c', s='d', data=data)

plt.xlabel('entry a')

plt.ylabel('entry b')

plt.show()
names = ['group_a', 'group_b', 'group_c']

values = [1, 10, 100]



plt.figure(figsize=(9, 3))



plt.subplot(131)

plt.bar(names, values)

plt.subplot(132)

plt.scatter(names, values)

plt.subplot(133)

plt.plot(names, values)

plt.suptitle('Categorical Plotting')

plt.show()