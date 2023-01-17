# Import modules|
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# If you decide to increase or decrease the font size in figures later, you can turn on the following line of code.
# plt.rcParams.update({'font.size': 16}) 

# Set the backend of matplotlib to the 'inline' backend so that plots appear below the code that produces them
# (See details at the bottom of this webpage: https://ipython.readthedocs.io/en/stable/interactive/plotting.html)
%matplotlib inline
x0 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
y0 = np.array([12.0, 7.3, 4.4, 2.5, 1.6, 1.0, 0.6, 0.4, 0.2])

# We can check the lengths of the x and y arrays match by using the command 'len()'.
# If the lengths of these arrays don't match, then we know we should check the data pairs.
print (len(x0), ',', len(y0))
#charing for R1 = (47 Ohms)
x = np.array([0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,84,87,90,93,96,99,102,105,108,111,114,117,120,123,126,129,132,135,138,141,144,147,150,153,156,159,162,165,168,171,174,177,180])
y = np.array([0.005,0.279,0.419,0.561,0.724,0.828,0.935,1.068,1.143,1.229,1.314,1.397,1.479,1.538,1.605,1.664,1.726,1.780,1.802,1.878,1.921,1.969,2.015,2.056,2.114,2.146,2.180,2.215,2.255,2.282,2.310,2.340,2.367,2.393,2.417,2.444,2.465,2.489,2.511,2.531,2.548,2.568,2.602,2.624,2.646,2.664,2.678,2.691,2.705,2.717,2.731,2.744,2.758,2.765,2.775,2.786,2.795,2.806,2.813, 2.821, 2.827])
print (len(x), ',', len(y))

#Dicharging for R1 = (47 Ohms)
x1 = x
y1 = np.array([2.690,2.535,2.438,2.320,2.217,2.123,2.039,1.958,1.873,1.793,1.722,1.648,1.570,1.506,1.444,1.375,1.318,1.257,1.206,1.150,1.098,1.052,1.007,0.955,0.915,0.875,0.837,0.798,0.759,0.724,0.692,0.659,0.627,0.600,0.573,0.544,0.514,0.488,0.460,0.437,0.418,0.395,0.380,0.361,0.345,0.323,0.312,0.297,0.283,0.270,0.259,0.247,0.236,0.225,0.215,0.205,0.195,0.187,0.180,0.172,0.165])
print (len(x1),',', len(y1))

#Charging for R2 = (100 Ohms)
x2 = np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,255,260,265,270,275,280,285,290,295,300,305,310,315,320,325,330,335,340,345,350,355,360])
y2 = np.array([0.006,0.190,0.324,0.448,0.570,0.678,0.784,0.878,0.966,1.055,1.133,1.211,1.281,1.348,1.415,1.475,1.532,1.590,1.641,1.694,1.742,1.787,1.833,1.875,1.918,1.956,1.993,2.030,2.064,2.097,2.130,2.160,2.191,2.218,2.246,2.272,2.297,2.322,2.345,2.367,2.389,2.411,2.430,2.450,2.468,2.486,2.503,2.520,2.535,2.552,2.566,2.580,2.595,2.608,2.621,2.634,2.645,2.657,2.668,2.679,2.690,2.700,2.710,2.719,2.728,2.737,2.746,2.754,2.762,2.770,2.777,2.784,2.791])
print (len(x2),',',len(y2))

#Discharging for R2 = 100 Ohms
x3 = x2
y3 = np.array([2.717,2.611,2.517,2.424,2.341,2.257,2.181,2.108,2.033,1.965,1.895,1.834,1.769,1.716,1.648,1.593,1.534,1.481,1.427,1.371,1.327,1.280,1.235,1.189,1.147,1.105,1.064,1.026,0.987,0.951,0.917,0.881,0.849,0.816,0.787,0.756,0.729,0.702,0.672,0.650,0.627,0.602,0.580,0.558,0.538,0.513,0.498,0.489,0.462,0.446,0.429,0.413,0.399,0.384,0.370,0.356,0.344,0.322,0.319,0.308,0.287,0.277,0.267,0.258,0.249,0.240,0.232,0.224,0.216,0.209, 0.209, 0.195, 0.190])
print (len(x3),',',len(y3))
# Plot data with blue circles at the data points (this is what 'bo' does - 'b' stands for blue and 'o' stands for circle)
#Charging a 45 Resistor capacitor 
plt.plot(x, y, 'bo')

# Attach labels and title (using LaTeX syntax)
plt.xlabel('$t$ (Seconds)')
plt.ylabel('$V$ (Volts)')
plt.title('Charging a Capacitor with 45 Ohm resistance')

# Display the figure
plt.show()
# For Discharging a 45 resitor capacitor 
plt.plot(x1, y1, 'bo')

# Attach labels and title (using LaTeX syntax)
plt.xlabel('$t$ (Seconds)')
plt.ylabel('$V$ (Volts)')
plt.title('Discharging a Capacitor with 45 Ohm resistance')

# Display the figure
plt.show()
# For charging a 100ohm resistor capacitor
plt.plot(x2, y2, 'bo')

# Attach labels and title (using LaTeX syntax)
plt.xlabel('$t$ (Seconds)')
plt.ylabel('$V$ (Volts)')
plt.title('Charging a Capacitor with 100 Ohm resistance')

# Display the figure
plt.show()
print (len(x2),',',len(y2))
#for discharging a 100 ohm capacitor 
plt.plot(x3, y3, 'bo')

# Attach labels and title (using LaTeX syntax)
plt.xlabel('$t$ (Seconds)')
plt.ylabel('$V$ (Volts)')
plt.title('Discharging a Capacitor with 100 Ohm resistance')

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
y_new1 = (1-(y/2.9)) #2.9 is used as Vmax 
plt.plot(x, np.log(y_new1), 'bd') 

# Attach labels and title 
plt.xlabel('$time$ (seconds)')
plt.ylabel('$\ln(V)$ (Volts)')
plt.title("Log of the Potenital Differces for 47 Ohms vs Time")

# Display the figure
plt.show()
plt.plot(x1, np.log(y1), 'bd') 

# Attach labels and title 
plt.xlabel('$x$ (unit)')
plt.ylabel('$\ln(y)$ (unit)')
plt.title("Log of the Potenital Differces for 47 Ohms vs Time")

# Display the figure
plt.show()
y_new2 = (1-(y2/2.9))
plt.plot(x2, np.log(y_new2), 'bd') 

# Attach labels and title 
plt.xlabel('$t$ (seconds)')
plt.ylabel('$\ln(V)$ (Volts)')
plt.title("Log of the Potenital Differces for 100 Ohms vs Time")

# Display the figure
plt.show()
print (len(x2),',',len(y_new2))
plt.plot(x3, np.log(y3), 'bd') 

# Attach labels and title 
plt.xlabel('$t$ (Seconds)')
plt.ylabel('$\ln(v)$ (Volts)')
plt.title("Log of the Potenital Differces for 100 Ohms vs Time")

# Display the figure
plt.show()
# Find the line of best fit for the data
slope, intercept, r_value, p_value, std_err = stats.linregress(x, np.log(y_new1))

# Create the line of best fit from the linear fit above
line0 = slope*x + intercept

# Plot the line of best fit in cyan and x vs ln(y) with blue diamonds
plt.plot(x, np.log(y_new1),'bd')
plt.plot(x, line0, 'r', label='y=-0.019169x + 0.327')

# Attach labels and title
plt.xlabel('$t$ (s)')
plt.ylabel('$\ln(ΔV_0 - ΔV)$ (V)')
plt.title('Log of the Charing Potenital Differences for 47 Ohm vs Time')

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
slope, intercept, r_value, p_value, std_err = stats.linregress(x1, np.log(y1))

# Create the line of best fit from the linear fit above
line = slope*x1 + intercept

# Plot the line of best fit in cyan and x vs ln(y) with blue diamonds
plt.plot(x1, np.log(y1),'bd')
plt.plot(x1, line, 'r', label='y=-0.01568x + 1.01')
# Attach labels and title
plt.xlabel('$t$ (s)')
plt.ylabel('$\ln(V)$ (V)')
plt.title('Log of the Discharing Potenital Differences for 47 Ohm vs Time')

# Add a grid to the plot
plt.grid(alpha=.4,linestyle='--')

# Show the legend in the plot
plt.legend()

# Display the figure
plt.show()

# Print the features from the linear fit
print(slope, intercept, r_value, p_value, std_err)
# Find the line of best fit for the data
slope, intercept, r_value, p_value, std_err = stats.linregress(x2, np.log(y_new2))

# Create the line of best fit from the linear fit above
line1 = slope*x2 + intercept

# Plot the line of best fit in cyan and x vs ln(y) with blue diamonds
plt.plot(x2, np.log(y_new2),'bd')
plt.plot(x2, line1, 'r', label='y= -0.008687x - 0.0324')
# Attach labels and title
plt.xlabel('$t$ (s)')
plt.ylabel('$\ln(ΔV_0 - ΔV)$ (V)')
plt.title('Log of the Charing Potenital Differences for 100 Ohm vs Time')

# Add a grid to the plot
plt.grid(alpha=.4,linestyle='--')

# Show the legend in the plot
plt.legend()

# Display the figure
plt.show()

# Print the features from the linear fit
print(slope, intercept, r_value, p_value, std_err)
print (len(x2),',',len(y_new2))
# Find the line of best fit for the data
slope, intercept, r_value, p_value, std_err = stats.linregress(x3, np.log(y3))

# Create the line of best fit from the linear fit above
line2 = slope*x3 + intercept

# Plot the line of best fit in cyan and x vs ln(y) with blue diamonds
plt.plot(x3, np.log(y3),'bd')
plt.plot(x3, line2, 'r', label='y= -0.00747x +1.02')
# Attach labels and title
plt.xlabel('$t$ (s)')
plt.ylabel('$\ln(V)$ (V)')
plt.title('Log of the Discharing Potenital Differences for 100 Ohm vs Time')

# Add a grid to the plot
plt.grid(alpha=.4,linestyle='--')

# Show the legend in the plot
plt.legend()

# Display the figure
plt.show()

# Print the features from the linear fit
print(slope, intercept, r_value, p_value, std_err)

