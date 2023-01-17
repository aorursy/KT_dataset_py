# Import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# If you decide to increase or decrease the font size in figures later, you can turn on the following line of code.
# plt.rcParams.update({'font.size': 16}) 

# Set the backend of matplotlib to the 'inline' backend so that plots appear below the code that produces them
# (See details at the bottom of this webpage: https://ipython.readthedocs.io/en/stable/interactive/plotting.html)
%matplotlib inline
# These are the data points for the 47ohm resistor (R1) when it is charging
t1c = np.array([1.34,1.8,2.42,3.25,4.36,5.86,7.87,10.57,14.19,19.05,25.58,34.35,46.12,61.93,83.15,111.65,149.92,201.3,270.3,362.94])
V1c = np.array([0.026,0.046,0.077,0.108,0.138,0.167,0.233,0.302,0.41,0.528,0.667,0.844,1.058,1.293,1.558,1.836,2.122,2.391,2.619,2.794])

# These are the data points for the 47ohm resistor (R1) when it is discharging
t1d = np.array([1.34,1.8,2.42,3.25,4.36,5.85,7.85,10.54,14.15,19,25.51,34.24,45.96,61.7,82.82,111.17,149.24,200.33,268.93,361])
V1d = np.array([2.717,2.712,2.67,2.649,2.628,2.607,2.571,2.514,2.439,2.363,2.257,2.12,1.958,1.753,1.509,1.223,0.922,0.625,0.374,0.194])

# Plotting the 47 ohm resistor charging
plt.figure('100 charging')
plt.plot(t1c, V1c, 'kx')

# Attach labels and title
plt.xlabel('$t$ (seconds)')
plt.ylabel('$V$ (volts)')
plt.title('Time-Voltage plot during charging when connected to 47$\Omega$ resistor')

# Display the figure
plt.show()


# Since the circuit is attached to a V0 = 3V battery we anticipate that it will have the equation V = V0*(1-exp(-t/tau)) meaning that if we take DV = V0 - V 
# we should get a decaying exponential just like during discharging
plt.figure('100 charging voltagediff')
# Define array to subtract V1c from
V0 = 3*np.ones(20)

DV1 = V0 - V1c



# Plotting this
plt.plot(t1c, DV1, 'kx')
plt.xlabel('$t$ (seconds)')
plt.ylabel('$\Delta V$ (volts)')
plt.title('Plot of $\Delta V = V_0 - V$ vs time during charging')

# Display the figure
plt.show()

# Plotting the 47 ohm resistor charging
plt.figure('47 discharging')
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
plt.plot(t1d, np.log(V1d), 'bx')

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
plt.plot(t1d, line1, 'b', label=Name1)
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

R1exact = 99.1

C = tau/R1exact

rR = 0.05

rC = np.sqrt(rR ** 2 + ((std_err + std_err1)) ** 2)

print(' The time constant is: ',tau,'\n','this gives a capacitance of: ',C, '\n','with relative error equal to: ',rC)