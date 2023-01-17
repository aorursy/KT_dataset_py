# Import modules

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats



# If you decide to increase or decrease the font size in figures later, you can turn on the following line of code.

# plt.rcParams.update({'font.size': 16}) 



# Set the backend of matplotlib to the 'inline' backend so that plots appear below the code that produces them

# (See details at the bottom of this webpage: https://ipython.readthedocs.io/en/stable/interactive/plotting.html)

%matplotlib inline
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

y = np.array([12.0, 7.3, 4.4, 2.5, 1.6, 1.0, 0.6, 0.4, 0.2])



# We can check the lengths of the x and y arrays match by using the command 'len()'.

# If the lengths of these arrays don't match, then we know we should check the data pairs.

print (len(x), ',', len(y))
# Plot data with blue circles at the data points (this is what 'bo' does - 'b' stands for blue and 'o' stands for circle)

plt.plot(x, y, 'bo')



# Attach labels and title (using LaTeX syntax)

plt.xlabel('$x$ (unit)')

plt.ylabel('$y$ (unit)')

plt.title('Title here')



# Display the figure

plt.show()
# Create arrays for the second data set and give these different names

x_new = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])

y_new = np.array([11.9, 9.8, 7.9, 6.6, 5.4, 4.4, 3.6, 3.0, 2.4, 2.0, 1.6, 1.3, 1.1, 0.9, 0.7, 0.6])



# Create a figure with a specific name 'Test figure' - this will allow us to refer to this particular figure later on.

plt.figure('Test figure')



# Plot the first data set with blue circles ('bo') and the second data set with red crosses ('rx')

# Add legend description for the two data sets

plt.plot(x, y, 'bo', label='Legend 1')

plt.plot(x_new, y_new, 'rx', label='Legend 2')



# Attach labels and title 

plt.xlabel('$x$ (unit)')

plt.ylabel('$y$ (unit)')

plt.title('Title here')



# Show the legend in the plot

plt.legend()



# Show a grid in the plot

plt.grid(alpha=.4,linestyle='--')



# Display the figure

plt.show()
# Set the figure to save

plt.figure('Test Figure') 



# Save the figure as a file in png, pdf or other formats

plt.savefig('TestFigure.png', bbox_inches='tight') # The output figure is saved to your project folder by default.

#plt.savefig('TestFigure.pdf', bbox_inches='tight')



# It's possible to save the figure in png by drag-and-drop: click on the output figure and drag it to a folder on your computer.
# Take the natural logarithm of y (this is what 'np.log(y)' does').

# Plot data with blue diamond markers at the data points (this is what 'bd' does)

# More on logarithms in numpy: https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.log.html

# More on marker types and colours: https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html

plt.plot(x, np.log(y), 'bd') 



# Attach labels and title 

plt.xlabel('$x$ (unit)')

plt.ylabel('$\ln(y)$ (unit)')

plt.title('Title here')



# Display the figure

plt.show()
# Find the line of best fit for the data

slope, intercept, r_value, p_value, std_err = stats.linregress(x, np.log(y))



# Create the line of best fit from the linear fit above

line = slope*x + intercept



# Plot the line of best fit in cyan and x vs ln(y) with blue diamonds

plt.plot(x, line, 'c', label='Equation here')

plt.plot(x, np.log(y),'bd')



# Attach labels and title

plt.xlabel('$x$ (unit)')

plt.ylabel('$\ln(y)$ (unit)')

plt.title('Title here')



# Add a grid to the plot

plt.grid(alpha=.4,linestyle='--')



# Show the legend in the plot

plt.legend()



# Display the figure

plt.show()



# Print the features from the linear fit

print(slope, intercept, r_value, p_value, std_err)



# After determining the slope and intercept, you can add the equation describing the linear model in the legend.
# Arrays for 1/d (x-axis)

inverse_d = np.array([1/0.000226, 1/0.000230, 1/0.000181, 1/0.000140, 1/0.000111])



# Arrays for change in y (y-axis)

delta_y = np.array([0.0034, 0.0039, 0.0048, 0.0061, 0.0080])



# Error bars

dx = 0.000002

dy = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0002])



# Create a figure with a specific name

plt.figure('DoubleSlitGraph')



# Plot the first data set with error bars

plt.errorbar(inverse_d, delta_y, xerr=dx, yerr=dy, fmt='.k')



# Attach labels and title 

plt.xlabel('1/d, m')

plt.ylabel('$delta$ y, m')

plt.title('1/d versus $delta$ y')



# Show a grid in the plot

plt.grid(alpha=.4,linestyle='--')



# Display the figure

plt.show()
# Arrays for 1/d (x-axis)

inverse_d2 = np.array([1/0.000226, 1/0.000181, 1/0.000140, 1/0.000111])

# Arrays for change in y (y-axis)

delta_y2 = np.array([0.0034, 0.0048, 0.0061, 0.0080])



dy2 = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0002])



# Find the line of best fit for the data

slope, intercept, r_value, p_value, std_err = stats.linregress(inverse_d2, delta_y2)



# Create the line of best fit from the linear fit above

line = slope*inverse_d2 + intercept



# Plot the line of best fit in cyan and x vs ln(y) with blue diamonds

plt.plot(inverse_d2, line, 'k', label='Line of best fit')

plt.plot(inverse_d2, delta_y2,'kx')

#plt.errorbar(inverse_d2, delta_y2, xerr=dx, yerr=dy2, fmt='.k')



# Attach labels and title

plt.xlabel('1/d, m')

plt.ylabel('$delta$ y, m')

plt.title('1/d versus $delta$ y')



# Add a grid to the plot

plt.grid(alpha=.4,linestyle='--')



# Show the legend in the plot

plt.legend()



# Display the figure

plt.show()



# Print the features from the linear fit

print(slope, intercept, r_value, p_value, std_err)



# After determining the slope and intercept, you can add the equation describing the linear model in the legend.