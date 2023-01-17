# Comments look like this line!

a = 3

a
import numpy as np

import scipy.stats as stats

# Now import matplotlib.pyplot

# Write down the gyromagnetic ratio of different elements



# Hydrogen



# Deuterium



# Carbon



# Oxygen



# Nitrogen



# Sulfur



# Fluorine
# This loads the data

calibration = np.load('../input/pythonforbeginners/calibration.npy')

dataset = np.load('../input/pythonforbeginners/dataset.npy')



calibrationFrequency = 5000000000

datasetFrequency = 2500000000
from scipy.fft import fft

from scipy.fft import fftfreq







# Plot the data!
def findStrength(data, sampleFreq):

    # Find the peak in the spectrum



    # Find the associated frequency



    # Calculate the strength of the magnet

    return None



magnetStrength = findStrength(calibration, calibrationFrequency)



magnetStrength