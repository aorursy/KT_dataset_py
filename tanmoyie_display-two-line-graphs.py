# Display two line graphs
# import necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Create the data table
time_series = np.matrix([[1, 10, 2], [2, 13, 4], [3, 12, 8], [4, 9, 14], [5, 10.5, 20]])

# Produce the figure
plt.figure(figsize=(12, 8)) # figure size
"""Draw the Earning of Person 1: 
X = Time, Column 0 of our time_series matrix
Y = Earning of Person 1, Column 1 of our time_series matrix
"""
plt.plot(time_series[:,0], time_series[:,1], 'b-', label = 'Person 1') 
plt.plot(time_series[:,0], time_series[:,2], 'r-', label = 'Person 2')
plt.xlabel('Date'); plt.ylabel('Earning'); plt.title('Earning of Person 1 & 2')
plt.legend();
# Reference: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matrix.html
