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
# Arrays for charging data sets.

x_charg_47 = np.array([0.00, 5.08, 10.11, 15.03, 20.03, 24.98, 30.04, 34.97, 40.04, 45.00, 50.01, 55.04, 59.97, 65.98, 70.00, 75.12, 80.02, 84.97, 90.04, 94.95, 100.02, 104.96, 110.00, 115.00, 120.02, 125.06, 129.96, 134.92, 140.01, 144.91, 150.08, 154.98, 160.13, 165.06, 169.97, 175.04, 180.02])

y_charg_47 = np.array([0.05, 0.381, 0.617, 0.820, 1.017, 1.174, 1.322, 1.447, 1.561, 1.670, 1.765, 1.857, 1.937, 2.011, 2.073, 2.145, 2.204, 2.251, 2.310, 2.356, 2.402, 2.441, 2.481, 2.516, 2.550, 2.570, 2.609, 2.637, 2.662, 2.685, 2.708, 2.728, 2.748, 2.767, 2.782, 2.799, 2.813])

x_charg_100 = np.array([0.00, 10.01, 19.98, 29.90, 40.00, 49.94, 59.83, 69.87, 79.95, 90.00, 99.84, 109.95, 119.94, 129.95, 140.02, 149.98, 160.02, 170.00, 179.90, 189.98, 199.96, 210.00, 219.98, 229.96, 240.00, 250.06, 260.03, 269.90, 280.10, 290.10, 299.98, 309.90, 320.12, 330.02, 339.96, 349.82, 359.96])

y_charg_100 = np.array([0.020, 0.316, 0.553, 0.774, 0.963, 1.130, 1.274, 1.408, 1.529, 1.639, 1.737, 1.823, 1.913, 1.991, 2.063, 2.129, 2.189, 2.244, 2.296, 2.344, 2.387, 2.430, 2.467, 2.502, 2.535, 2.566, 2.594, 2.620, 2.645, 2.668, 2.689, 2.709, 2.728, 2.745, 2.762, 2.777, 2.791])



# Create a figure with a specific name

plt.figure('ChargingTheCapacitor')



# Plot the first data set with blue crosses ('bx') and the second data set with red crosses ('rx')

# Add legend description for the two data sets

plt.plot(x_charg_47, y_charg_47, 'bx', label = '47 Ohm Resistor')

plt.plot(x_charg_100, y_charg_100, 'rx', label='100 Ohm Resistor')



# Attach labels and title 

plt.xlabel('Time, s')

plt.ylabel('Voltage, V')

plt.title('Capacitor Voltage over Time - Charging')



# Show the legend in the plot

plt.legend()



# Show a grid in the plot

plt.grid(alpha=.4,linestyle='--')



# Display the figure

plt.show()
# Arrays for discharging data sets.

x_dis_47 = np.array([0.00, 5.05, 10.02, 14.96, 20.05, 25.04, 29.97, 34.95, 39.98, 45.08, 49.97, 55.02, 59.94, 64.93, 69.96, 74.96, 79.95, 84.97, 90.12, 95.00, 100.00, 105.02, 109.94, 115.06, 119.96, 124.94, 130.04, 135.04, 140.02, 145.00, 149.97, 155.06, 160.00, 165.02, 170.08, 175.04, 180.00])

y_dis_47 = np.array([2.690, 2.456, 2.283, 2.132, 1.978, 1.846, 1.712, 1.598, 1.498, 1.379, 1.282, 1.196, 1.102, 1.022, 0.944, 0.875, 0.811, 0.747, 0.689, 0.638, 0.590, 0.544, 0.504, 0.464, 0.432, 0.398, 0.367, 0.340, 0.315, 0.291, 0.270, 0.250, 0.232, 0.215, 0.199, 0.185, 0.172])

x_dis_100 = np.array([0.00, 9.95, 19.95, 30.08, 39.91, 50.10, 59.92, 69.96, 80.06, 90.01, 100.02, 110.00, 119.96, 130.01, 140.06, 150.00, 160.00, 170.00, 179.95, 190.08, 200.04, 209.90, 220.08, 230.00, 240.20, 250.10, 259.92, 270.00, 280.06, 290.06, 300.05, 309.94, 319.93, 330.05, 340.00, 349.84, 359.97])

y_dis_100 = np.array([2.717, 2.517, 2.341, 2.171, 2.033, 1.895, 1.769, 1.648, 1.534, 1.427, 1.327, 1.233, 1.147, 1.064, 0.987, 0.915, 0.847, 0.787, 0.729, 0.675, 0.625, 0.580, 0.537, 0.497, 0.462, 0.429, 0.399, 0.369, 0.343, 0.319, 0.297, 0.277, 0.258, 0.240, 0.224, 0.207, 0.195])



# Create a figure with a specific name

plt.figure('DischargingTheCapacitor')



# Plot the first data set with blue crosses ('bx') and the second data set with red crosses ('rx')

# Add legend description for the two data sets

plt.plot(x_dis_47, y_dis_47, 'bx', label = '47 Ohm Resistor')

plt.plot(x_dis_100, y_dis_100, 'rx', label='100 Ohm Resistor')



# Attach labels and title 

plt.xlabel('Time, s')

plt.ylabel('Voltage, V')

plt.title('Capacitor Voltage over Time - Discharging')



# Show the legend in the plot

plt.legend()



# Show a grid in the plot

plt.grid(alpha=.4,linestyle='--')



# Display the figure

plt.show()
# Set the figure to save

plt.figure('Charging the Capacitor')

plt.figure('Discharging the Capacitor')



# Save the figure as a file in png, pdf or other formats

#plt.savefig('TestFigure.png', bbox_inches='tight') The output figure is saved to your project folder by default.

#plt.savefig('TestFigure.pdf', bbox_inches='tight')



plt.savefig('ChargingTheCapacitor.pdf', bbox_inches = 'tight')

plt.savefig('DischargingTheCapacitor.pdf', bbox_inches = 'tight')



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
plt.plot(x_charg_47, np.log(3.000-(y_charg_47)), 'bd', label = '47 Ohm Resistor')

plt.plot(x_charg_100, np.log(3.000-(y_charg_100)), 'rd', label='100 Ohm Resistor')



# Attach labels and title 

plt.xlabel('Time, s')

plt.ylabel('ln(Volatge), V')

plt.title('ln(Voltage) over Time - Charging')



# Show the legend in the plot

plt.legend()



# Show a grid in the plot

plt.grid(alpha=.4,linestyle='--')



# Display the figure

plt.show()
plt.plot(x_dis_47, np.log(y_dis_47), 'bd', label = '47 Ohm Resistor')

plt.plot(x_dis_100, np.log(y_dis_100), 'rd', label='100 Ohm Resistor')



# Attach labels and title 

plt.xlabel('Time, s')

plt.ylabel('ln(Volatge), V')

plt.title('ln(Voltage) over Time - Discharging')



# Show the legend in the plot

plt.legend()



# Show a grid in the plot

plt.grid(alpha=.4,linestyle='--')



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
# Find the line of best fit for the data

slope, intercept, r_value, p_value, std_err = stats.linregress(x_charg_47, np.log(3.000-(y_charg_47)))



# Create the line of best fit from the linear fit above

line = slope*x_charg_47 + intercept



# Plot the line of best fit in cyan and x vs ln(y) with blue diamonds

plt.plot(x_charg_47, line, 'c', label='$\ln(V_{0} - V) = \ln(V_{0}) - t/tau$: 100 Ohm')

plt.plot(x_charg_47, np.log(3.000-(y_charg_100)),'bd')



# Attach labels and title

plt.xlabel('Time, s')

plt.ylabel('ln(Volatge), V')

plt.title('Linearlised 47 Ohm: ln(Voltage) over Time - Charging')



# Add a grid to the plot

plt.grid(alpha=.4,linestyle='--')



# Show the legend in the plot

plt.legend()



# Display the figure

plt.show()



# Print the features from the linear fit

print(slope, intercept, r_value, p_value, std_err)



# After determining the slope and intercept, you can add the equation describing the linear model in the legend.
# Find the line of best fit for the data

slope, intercept, r_value, p_value, std_err = stats.linregress(x_charg_100, np.log(3.000-(y_charg_100)))



# Create the line of best fit from the linear fit above

line = slope*x_charg_100 + intercept



# Plot the line of best fit in cyan and x vs ln(y) with blue diamonds

plt.plot(x_charg_100, line, 'm', label='$\ln(V_{0} - V) = \ln(V_{0}) - t/tau$: 100 Ohm')

plt.plot(x_charg_100, np.log(3.000-(y_charg_100)),'rd')



# Attach labels and title

plt.xlabel('Time, s')

plt.ylabel('ln(Volatge), V')

plt.title('Linearlised 100 Ohm: ln(Voltage) over Time - Charging')



# Add a grid to the plot

plt.grid(alpha=.4,linestyle='--')



# Show the legend in the plot

plt.legend()



# Display the figure

plt.show()



# Print the features from the linear fit

print(slope, intercept, r_value, p_value, std_err)



# After determining the slope and intercept, you can add the equation describing the linear model in the legend.
# Find the line of best fit for the data

slope, intercept, r_value, p_value, std_err = stats.linregress(x_dis_47, np.log(y_dis_47))



# Create the line of best fit from the linear fit above

line = slope*x_dis_47 + intercept



# Plot the line of best fit in cyan and x vs ln(y) with blue diamonds

plt.plot(x_dis_47, line, 'c', label='$\ln(V) = \ln(V_{0}) - t/tau$: 47 Ohm')

plt.plot(x_dis_47, np.log(y_dis_47),'bd')



# Attach labels and title

plt.xlabel('Time, s')

plt.ylabel('ln(Volatge), V')

plt.title('Linearlised 47 Ohm: ln(Voltage) over Time - Discharging')



# Add a grid to the plot

plt.grid(alpha=.4,linestyle='--')



# Show the legend in the plot

plt.legend()



# Display the figure

plt.show()



# Print the features from the linear fit

print(slope, intercept, r_value, p_value, std_err)



# After determining the slope and intercept, you can add the equation describing the linear model in the legend.
# Find the line of best fit for the data

slope, intercept, r_value, p_value, std_err = stats.linregress(x_dis_100, np.log(y_dis_100))



# Create the line of best fit from the linear fit above

line = slope*x_dis_100 + intercept



# Plot the line of best fit in cyan and x vs ln(y) with blue diamonds

plt.plot(x_dis_100, line, 'm', label='$\ln(V) = \ln(V_{0}) - t/tau$: 100 Ohm')

plt.plot(x_dis_100, np.log(y_dis_100),'rd')



# Attach labels and title

plt.xlabel('Time, s')

plt.ylabel('ln(Volatge), V')

plt.title('Linearlised 100 Ohm: ln(Voltage) over Time - Discharging')



# Add a grid to the plot

plt.grid(alpha=.4,linestyle='--')



# Show the legend in the plot

plt.legend()



# Display the figure

plt.show()



# Print the features from the linear fit

print(slope, intercept, r_value, p_value, std_err)



# After determining the slope and intercept, you can add the equation describing the linear model in the legend.