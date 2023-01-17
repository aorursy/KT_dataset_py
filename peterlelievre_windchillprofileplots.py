%matplotlib inline

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import math

# Set font size:

matplotlib.rcParams.update({'font.size': 16})

# Define the bivariate windchill temperature function ...

def f(a,w): # a is ambient temperature and w is wind speed

    return 13.12 + 0.6215*a - 11.37*w**0.16 + 0.3965*a*w**0.16
# Set up the plotting area (figure) and make it a little larger than the default size:

plt.figure(figsize=(10,10))



# Define the wind speed values:

ww = np.linspace(0,30,100) # wind speed values

# x=np.linspace(A,B,N) defines the x data points as N points equally spaced between A and B



# Set which ambient temperature values to show profiles of:

aa = np.linspace(-10,10,5) # ambient temperature values



# Plot randomly coloured lines for each profile:

for a in aa: # (loop over each ambient temperature)

    tt = f(a,ww) # windchill temperatures calculated for the current ambient temperature and all windspeeds

    plt.plot(ww,tt)



# Add labels on the axes:

plt.xlabel('Wind speed (km/h)')

plt.ylabel('Windchill Temperature (C)')



# Add a legend:

plt.legend(aa,title='Ambient Temperature (C)');
