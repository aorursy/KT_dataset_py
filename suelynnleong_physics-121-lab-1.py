# Import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# If you decide to increase or decrease the font size in figures later, you can turn on the following line of code.
# plt.rcParams.update({'font.size': 16}) 

# Set the backend of matplotlib to the 'inline' backend so that plots appear below the code that produces them
# (See details at the bottom of this webpage: https://ipython.readthedocs.io/en/stable/interactive/plotting.html)
%matplotlib inline
t_47 = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0])
Vc_47 = np.array([0.004, 1.007, 1.561, 1.940, 2.207, 2.404, 2.550, 2.663, 2.749, 2.815])
Vd_47 = np.array([2.672, 1.972, 1.477, 1.093, 0.807, 0.586, 0.426, 0.313, 0.230, 0.171])

t_100 = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 240.0, 260.0, 280.0, 300.0, 320.0, 340.0, 360.0])
Vc_100 = np.array([0.026, 0.563, 0.966, 1.281, 1.532, 1.740, 1.918, 2.064, 2.191, 2.297, 2.388, 2.468, 2.535, 2.594, 2.645, 2.689, 2.728, 2.762, 2.791])
Vd_100 = np.array([2.717, 2.338, 2.030, 1.766, 1.532, 1.325, 1.144, 0.987, 0.847, 0.727, 0.625, 0.538, 0.462, 0.398, 0.343, 0.297, 0.257, 0.224, 0.195])

# We can check the lengths of the x and y arrays match by using the command 'len()'.
# If the lengths of these arrays don't match, then we know we should check the data pairs.
print (len(t_47), ',', len(Vc_47), ',' , len(Vd_47))
print (len(t_100), ',', len(Vc_100), ',' , len(Vd_100))
t_u = 0.4
V_u = 0.0005
# Plot data with blue circles at the data points (this is what 'bo' does - 'b' stands for blue and 'o' stands for circle)
plt.plot(t_47, Vc_47, 'bo')

# Attach labels and title (using LaTeX syntax)
plt.xlabel('$Time$ (s)')
plt.ylabel('$Voltage$ (V)')
plt.title('Charging with 47$\Omega$ Resistor')

# Display the figure
plt.show()
# Create a figure with a specific name 'Test figure' - this will allow us to refer to this particular figure later on.
plt.figure('Discharging')

# Plot the points with error bars
plt.errorbar(t_47, Vd_47, yerr=V_u, xerr=t_u, fmt='o', c='darkviolet', markersize=6, label='$47.4\Omega$')
plt.errorbar(t_100, Vd_100, yerr=V_u, xerr=t_u, fmt='*', c='crimson', markersize=10, label='$99.1\Omega$')

# Attach labels and title 
plt.xlabel('$Time, t$ (s)')
plt.ylabel('$Voltage, V$ (V)')
plt.title('Discharging, Voltage vs Time')

# Show the legend in the plot
plt.legend()

# Show a grid in the plot
plt.grid(alpha=.4,linestyle='--')

# Display the figure
plt.show()
# Create a figure with a specific name 'Test figure' - this will allow us to refer to this particular figure later on.
plt.figure('Charging')

# Plot points with error bars
plt.errorbar(t_47, Vc_47, yerr=V_u, xerr=t_u, fmt='o', c='blueviolet', markersize=6, label='$47.4\Omega$')
plt.errorbar(t_100, Vc_100, yerr=V_u, xerr=t_u, fmt='*', c='firebrick', markersize=10, label='$99.1\Omega$')

# Attach labels and title 
plt.xlabel('$Time, t$ (s)')
plt.ylabel('$Voltage, V$ (V)')
plt.title('Charging, Voltage vs Time')

# Show the legend in the plot
plt.legend()

# Show a grid in the plot
plt.grid(alpha=.4,linestyle='--')

# Display the figure
plt.show()
# Create a figure with a specific name 'Test figure' - this will allow us to refer to this particular figure later on.
plt.figure('Discharging, log')

#Plot points with error bars
plt.errorbar(t_47, np.log(Vd_47), yerr=V_u, xerr=t_u, fmt='o', c='darkviolet', markersize=6)
plt.errorbar(t_100, np.log(Vd_100), yerr=V_u, xerr=t_u, fmt='*', c='crimson', markersize=10)

# LOBF
slope, intercept, r_value, p_value, std_err = stats.linregress(t_47, np.log(Vd_47))
line = slope*t_47 + intercept
plt.plot(t_47, line, c='mediumorchid', label='$47.4\Omega: ln(V) = -0.0154 \: t + 0.9965$')

# LOBF
slope, intercept, r_value, p_value, std_err = stats.linregress(t_100, np.log(Vd_100))
line = slope*t_100 + intercept
plt.plot(t_100, line, c='hotpink', label='$99.1\Omega: ln(V) = -0.0074 \: t + 1.0086$')

# Attach labels and title 
plt.xlabel('$Time, t$ (s)')
plt.ylabel('$ln(Voltage), lnV$ ($ln$V)')
plt.title('Discharging, ln(Voltage) vs Time')

# Show the legend in the plot
plt.legend()

# Show a grid in the plot
plt.grid(alpha=.4,linestyle='--')

# Display the figure
plt.show()

#print values for line
print(slope, intercept, std_err)
# Create a figure with a specific name 'Test figure' - this will allow us to refer to this particular figure later on.
plt.figure('Charging, log')

# Plot points with error bars
plt.errorbar(t_47, np.log(3-Vc_47), yerr=V_u, xerr=t_u, fmt='o', c='blueviolet', markersize=6)
plt.errorbar(t_100, np.log(3-Vc_100), yerr=V_u, xerr=t_u, fmt='*', c='firebrick', markersize=10)

# LOBF 47
slope, intercept, r_value, p_value, std_err = stats.linregress(t_47, np.log(3-Vc_47))
line = slope*t_47 + intercept
plt.plot(t_47, line, c='darkorchid', label='$47.4\Omega: ln(3-V) = -0.0151t + 1.0053$')
print(slope, intercept, std_err)

# LOBF 100
slope, intercept, r_value, p_value, std_err = stats.linregress(t_100, np.log(3-Vc_100))
line = slope*t_100 + intercept
plt.plot(t_100, line, c='palevioletred', label='$99.1\Omega: ln(3-V) = -0.0072 t + 0.9856$')
print(slope, intercept, std_err)

# Attach labels and title 
plt.xlabel('$Time, t$ (s)')
plt.ylabel('$ln(3-Voltage), lnV$ ($ln$V)')
plt.title('Charging, ln(3-Voltage) vs Time')

# Show the legend in the plot
plt.legend()

# Show a grid in the plot
plt.grid(alpha=.4,linestyle='--')

# Display the figure
plt.show()


# Set the figure to save
plt.figure('Discharging') 

# Save the figure as a file in png, pdf or other formats
plt.savefig('Charging.jpg', bbox_inches='tight') # The output figure is saved to your project folder by default.
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