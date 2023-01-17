# Import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random as r

# If you decide to increase or decrease the font size in figures later, you can turn on the following line of code.
# plt.rcParams.update({'font.size': 16}) 

# Set the backend of matplotlib to the 'inline' backend so that plots appear below the code that produces them
# (See details at the bottom of this webpage: https://ipython.readthedocs.io/en/stable/interactive/plotting.html)
%matplotlib inline
# These are the data points for the 47ohm resistor (R1) when it is charging
t1c = np.array([1.3,1.69,2.19,2.84,3.69,4.79,6.22,8.07,10.47,13.6,17.65,22.92,29.75,38.63,50.15,65.1,84.52,109.72,142.44,184.92])
V1c = np.array([0.106,0.138,0.17,0.189,0.232,0.291,0.392,0.497,0.599,0.733,0.886,1.079,1.278,1.508,1.755,2.003,2.244,2.468,2.669,2.824])

# These are the data points for the 47ohm resistor (R1) when it is discharging
t1d = np.array([1.3,1.68,2.18,2.83,3.68,4.77,6.19,8.04,10.43,13.53,17.55,22.78,29.56,38.35,49.76,64.57,83.78,108.71,141.06,183.03])
V1d = np.array([2.599,2.577,2.564,2.526,2.515,2.475,2.415,2.362,2.266,2.175,2.053,1.916,1.734,1.522,1.292,1.03,0.765,0.516,0.312,0.165,])

# Plotting the 47 ohm resistor charging
plt.figure('47O charging')
plt.plot(t1c, V1c, 'kx')

# Attach labels and title
plt.xlabel('$t$ (seconds)')
plt.ylabel('$V$ (volts)')
plt.title('Time-Voltage plot during charging when connected to 47$\Omega$ resistor')

# Display the figure
plt.show()


# Since the circuit is attached to a V0 = 3V battery we anticipate that it will have the equation V = V0*(1-exp(-t/tau)) meaning that if we take DV = V0 - V 
# we should get a decaying exponential just like during discharging

# Define array to subtract V1c from
V0 = 3*np.ones(20)

DV1 = V0 - V1c



# Plotting this
plt.plot(t1c, DV1, 'kx')
plt.xlabel('$t$ (seconds)')
plt.ylabel('$\Delta V$ (volts)')
plt.title('Plot of $\Delta V = V_0 - V$ vs time')

# Display the figure
plt.show()

# Plotting the 47 ohm resistor charging
plt.figure('47O discharging')
plt.plot(t1d, V1d, 'kx')

# Attach labels and title 
plt.xlabel('$t$ (seconds)')
plt.ylabel('$V$ (volts)')
plt.title('Time-Voltage plot during discharging when connected to 47$\Omega$ resistor')

# Display the figure
plt.show()

# Set the figure to save
plt.figure('Test Figure') 

# Save the figure as a file in png, pdf or other formats
plt.savefig('TestFigure.png', bbox_inches='tight') # The output figure is saved to your project folder by default.
#plt.savefig('TestFigure.pdf', bbox_inches='tight')

# It's possible to save the figure in png by drag-and-drop: click on the output figure and drag it to a folder on your computer.
# Here we plot the natural log of DV = V0-V against time during the charging of the capacitor, this should give us a straight line whose 
# gradient is the inverse of the time constant

# Plot Logs against time
plt.plot(t1c, np.log(DV1), 'rx')


# Attach labels and title 
plt.xlabel('$t$ (seconds)')
plt.ylabel('$\ln(\Delta V)$ ')
plt.title('$ln(\Delta V)$ vs $t$ during charging')

# Display the figure
plt.show()

# Doing the same for discharging
plt.plot(t1d, np.log(V1d), 'rx')

# Attach labels and title 
plt.xlabel('$t$ (seconds)')
plt.ylabel('$\ln(V)$ ')
plt.title('$\ln(V)$ vs $t$ during discharging')

# Display the figure
plt.show()

# Find the line of best fit for the data
slope, intercept, r_value, p_value, std_err = stats.linregress(t1d, np.log(DV1))

print(slope, intercept, std_err)
Name = str(np.round(slope,5))+'*t+'+str(np.round(intercept,5))

# Create the line of best fit from the linear fit above
line = slope*t1d + intercept

# Plot the line of best fit 
plt.plot(t1d, line, 'r', label=Name)
plt.plot(t1d, np.log(DV1),'kd')

# Attach labels and title
plt.xlabel('$t$ (seconds)')
plt.ylabel('$\ln(\Delta V)$ ')
plt.title('$\ln(\Delta V)$ vs $t$ during discharging')

# Add a grid to the plot
plt.grid(alpha=.4,linestyle='--')

# Show the legend in the plot
plt.legend()

# Display the figure
plt.show()


# Find the line of best fit for the data
slope1, intercept1, r_value, p_value, std_err1 = stats.linregress(t1d, np.log(V1d))

print(slope1, intercept1, std_err1)
Name1 = str(np.round(slope1,5))+'*t+'+str(np.round(intercept1,5))

# Create the line of best fit from the linear fit above
line1 = slope1*t1d + intercept1

# Plot the line of best fit 
plt.plot(t1d, line1, 'r', label=Name1)
plt.plot(t1d, np.log(V1d),'kd')

# Attach labels and title
plt.xlabel('$t$ (seconds)')
plt.ylabel('$\ln(V)$ ')
plt.title('$\ln(V)$ vs $t$ during discharging')

# Add a grid to the plot
plt.grid(alpha=.4,linestyle='--')

# Show the legend in the plot
plt.legend()

# Display the figure
plt.show()


slope_av = 0.5 * (slope+slope1)

tau = -1/slope_av

R1exact = 47.4

C = tau/R1exact

rR = 0.05

rC = np.sqrt(rR ** 2 + (0.5*(std_err + std_err1)) ** 2)

print(tau, '\n'
     ,C, '\n'
     ,rC)