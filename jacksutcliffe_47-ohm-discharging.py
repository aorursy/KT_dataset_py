# Import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# If you decide to increase or decrease the font size in figures later, you can turn on the following line of code.
# plt.rcParams.update({'font.size': 16}) 

# Set the backend of matplotlib to the 'inline' backend so that plots appear below the code that produces them
# (See details at the bottom of this webpage: https://ipython.readthedocs.io/en/stable/interactive/plotting.html)
%matplotlib inline
#y = np.array([2.661, 2.456, 2.283, 2.132, 1.972, 1.827, 1.716, 1.59, 1.477, 1.375, 1.273, 1.183, 1.102, 1.05, 0.936, 0.873, 0.8, 0.735, 0.695, 0.638, 0.587, 0.544, 0.497, 0.463, 0.428, 0.396, 0.366, 0.34, 0.316, 0.291, 0.27, 0.249, 0.232, 0.214, 0.198, 0.185, 0.172])
y = np.array([0, 0.365, 0.617, 0.84, 1.024, 1.174, 1.328, 1.452, 1.568, 1.67, 1.765, 1.857, 1.94, 2.023, 2.094, 2.194, 2.209, 2.265, 2.305, 2.36, 2.4, 2.445, 2.491, 2.521, 2.553, 2.582, 2.61, 2.641, 2.659, 2.686, 2.708, 2.728, 2.749, 2.766, 2.784, 2.799, 2.815])
x = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180])

# We can check the lengths of the x and y arrays match by using the command 'len()'.
# If the lengths of these arrays don't match, then we know we should check the data pairs.
print (len(x), ',', len(y))
# Plot data with blue circles at the data points (this is what 'bo' does - 'b' stands for blue and 'o' stands for circle)
plt.plot(x, y, 'bx')

# Attach labels and title (using LaTeX syntax)
plt.xlabel('$x$ (Time s )')
plt.ylabel('$y$ (Potential Difference ∆V)')
plt.title('47 Ω Capacitor Charging ')

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
z = np.array([1, 1.440514008, 1.853359615, 2.316366977, 2.784309758, 3.234906419, 3.773488858, 4.271649276, 4.797044504, 5.312167797, 5.841572358, 6.404494439, 6.958750971, 7.560973865, 8.117319591, 8.971025544, 9.106605233, 9.631124601, 10.02417825, 10.59095145, 11.02317638, 11.5305496, 12.07334343, 12.44103148, 12.84558278, 13.22355885, 13.59905085, 14.02722382, 14.28199996, 14.67286691, 14.999247, 15.30225189, 15.62699707, 15.89492697, 16.18362616, 16.42821034, 16.69317578])
slope, intercept, r_value, p_value, std_err = stats.linregress(x, np.exp(y))

# Create the line of best fit from the linear fit above
line = slope*x + intercept

# Plot the line of best fit in cyan and x vs ln(y) with blue diamonds
plt.plot(x, line, 'c', label='Equation here')
plt.plot(x, np.exp(y),'rx')

# Attach labels and title
plt.xlabel('$x$ (Time )')
plt.ylabel('$\exp(y)$ (Potential Difference ∆V)')
plt.title(' 47 Ohm Charging exp(∆V) vs Time  ')

# Add a grid to the plot
plt.grid(alpha=.4,linestyle='--')

# Show the legend in the plot
plt.legend()

# Display the figure
plt.show()

# Print the features from the linear fit
print(slope, intercept, r_value, p_value, std_err)

# After determining the slope and intercept, you can add the equation describing the linear model in the legend.