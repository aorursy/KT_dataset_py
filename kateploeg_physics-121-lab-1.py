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
# Create arrays for the second data set and give these different names

x = np.array([0.20, 20.20, 40.60, 60.38, 80.40, 100.43, 120.50, 140.55, 160.33, 180.34])

y = np.array([0.086, 1.024, 1.573, 1.940, 2.209, 2.406, 2.552, 2.664, 2.749, 2.815])



x_new = np.array([0.38, 40.40, 80.47, 120.40, 160.48, 200.55, 240.60, 280.22, 320.50, 360.48])

y_new = np.array([0.046, 0.972, 1.533, 1.918, 2.191, 2.391, 2.537, 2.645, 2.729, 2.792])



# Create a figure with a specific name 'Test figure' - this will allow us to refer to this particular figure later on.

plt.figure('Test figure')



# Plot the first data set with blue circles ('bo') and the second data set with red crosses ('rx')

# Add legend description for the two data sets

plt.plot(x, y, 'bo', label='47Ω Resistor')

plt.plot(x_new, y_new, 'rx', label='100Ω Resistor')



# Attach labels and title 

plt.xlabel('$Time$ (s)')

plt.ylabel('$Voltage$ (V)')

plt.title('Charging the 1F Capacitor - Before Linearization')



# Show the legend in the plot

plt.legend()



# Show a grid in the plot

plt.grid(alpha=.4,linestyle='--')



# Display the figure

plt.show()
# Create arrays for the second data set and give these different names

x = np.array([0.00, 20.46, 40.44, 60.26, 80.43, 100.10, 120.31, 140.32, 160.55, 180.27])

y = np.array([2.735, 1.964, 1.477, 1.098, 0.804, 0.597, 0.428, 0.313, 0.230, 0.172])



x_new = np.array([0.35, 40.14, 80.34, 120.26, 160.40, 200.36, 240.58, 280.36, 320.53, 360.48])

y_new = np.array([2.712, 2.030, 1.529, 1.144, 0.846, 0.624, 0.461, 0.342, 0.256, 0.195])



# Create a figure with a specific name 'Test figure' - this will allow us to refer to this particular figure later on.

plt.figure('Test figure')



# Plot the first data set with blue circles ('bo') and the second data set with red crosses ('rx')

# Add legend description for the two data sets

plt.plot(x, y, 'bo', label='47Ω Resistor')

plt.plot(x_new, y_new, 'rx', label='100Ω Resistor')



# Attach labels and title 

plt.xlabel('$Time$ (s)')

plt.ylabel('$Voltage$ (V)')

plt.title('Discharging the 1F Capacitor - Before Linearization')



# Show the legend in the plot

plt.legend()



# Show a grid in the plot

plt.grid(alpha=.4,linestyle='--')



# Display the figure

plt.show()
# Create arrays for the second data set and give these different names

x = np.array([0.20, 20.20, 40.60, 60.38, 80.40, 100.43, 120.50, 140.55, 160.33, 180.34])

y = np.array([0.086, 1.024, 1.573, 1.940, 2.209, 2.406, 2.552, 2.664, 2.749, 2.815])



x_new = np.array([0.38, 40.40, 80.47, 120.40, 160.48, 200.55, 240.60, 280.22, 320.50, 360.48])

y_new = np.array([0.046, 0.972, 1.533, 1.918, 2.191, 2.391, 2.537, 2.645, 2.729, 2.792])



# Show the legend in the plot

plt.legend()



# Show a grid in the plot

plt.grid(alpha=.4,linestyle='--')



plt.plot(x, np.log(3-y), 'bd', label='47Ω Resistor')                          # doing 3-y to represent (ΔVnought-ΔV) for charging

plt.plot(x_new, np.log(3-y_new), 'rx', label='100Ω Resistor')



# Attach labels and title 

plt.xlabel('$Time$ (s)')

plt.ylabel('$\ln(ΔV)$ (V)')

plt.title('Linearized Charging of the 1F Capacitor')



# Find the line of best fit for the data

slope, intercept, r_value, p_value, std_err = stats.linregress(x, np.log(3-y))

slope_new, intercept_new, r_value_new, p_value_new, std_err_new = stats.linregress(x_new, np.log(3-y_new))



# Create the line of best fit from the linear fit above

line = slope*x + intercept



# Plot the line of best fit in cyan and x vs ln(y) with blue diamonds

plt.plot(x, line, 'b', label='ln(ΔV) = -0.0149t + 1.00')

plt.plot(x, np.log(3-y),'bo')



plt.plot(x_new, line, 'r', label='ln(ΔV) = -0.00726t + 1.00')

plt.plot(x_new, np.log(3-y_new),'rx')



plt.legend()



# Print the features from the linear fit

print("(x,y) = ", slope, intercept, r_value, p_value, std_err)

print("(x_new, y_new) = ", slope_new, intercept_new, r_value_new, p_value_new, std_err_new)



# Display the figure

plt.show()
# Create arrays for the second data set and give these different names

x = np.array([0.00, 20.46, 40.44, 60.26, 80.43, 100.10, 120.31, 140.32, 160.55, 180.27])

y = np.array([2.735, 1.964, 1.477, 1.098, 0.804, 0.597, 0.428, 0.313, 0.230, 0.172])



x_new = np.array([0.35, 40.14, 80.34, 120.26, 160.40, 200.36, 240.58, 280.36, 320.53, 360.48])

y_new = np.array([2.712, 2.030, 1.529, 1.144, 0.846, 0.624, 0.461, 0.342, 0.256, 0.195])



# Plot the first data set with blue circles ('bo') and the second data set with red crosses ('rx')

# Add legend description for the two data sets

plt.plot(x, np.log(y), 'bo', label='47Ω Resistor')

plt.plot(x_new, np.log(y_new), 'rx', label='100Ω Resistor')



# Attach labels and title 

plt.xlabel('$Time$ (s)')

plt.ylabel('$ln(ΔV)$ (V)')

plt.title('Discharging the 1F Capacitor - After Linearization')



# Show the legend in the plot

plt.legend()



# Find the line of best fit for the data

slope, intercept, r_value, p_value, std_err = stats.linregress(x, np.log(y))

slope_new, intercept_new, r_value_new, p_value_new, std_err_new = stats.linregress(x_new, np.log(y_new))



# Create the line of best fit from the linear fit above

line = slope*x + intercept



# Plot the line of best fit in cyan and x vs ln(y) with blue diamonds

plt.plot(x, line, 'b', label='ln(ΔV) = -0.0154t + 1.01')

plt.plot(x, np.log(y),'bo')



plt.plot(x_new, line, 'r', label='ln(ΔV) = -0.00737t + 1.01')

plt.plot(x_new, np.log(y_new),'rx')



plt.legend()



# Print the features from the linear fit

print(slope, intercept, r_value, p_value, std_err)

print(slope_new, intercept_new, r_value_new, p_value_new, std_err_new)



# Show a grid in the plot

plt.grid(alpha=.4,linestyle='--')



# Display the figure

plt.show()