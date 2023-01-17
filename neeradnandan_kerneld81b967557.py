import numpy as np

import matplotlib.pyplot as plt

import pandas as pd





# Read the text file and put each entry into the two array variables dist and Mv

# it skips the first two rows are not the data

mv_sun=np.loadtxt('../input/Mv_stars_ini.txt',skiprows=2,unpack=True)

mv,dist=np.loadtxt('../input/MvDistLimits.txt',unpack=True)



# Compute a histogram of distances, with a total of 10 equally spaced bins

# Returns the alues of the histograms and the edge alue of each bin



plt.hist(mv_sun, bins=10, color='green')



# Plot the histogram



plt.show()



# In order to fit the y(x) with a straight line you can use polyfit

# The param variable will store the two parameters a and b of y=ax+b

#param=np.polytfit(x,y,1)





# you can perform any operations  on any arrays:

# example: DistKpc=dist/1.e3

# you can multiply/divide/add arrays together as long as they have the same sizes

# Do not get stuck because of python, if you have a python issue ask around
mv_sun
import os

print(os.listdir('../input'))
def emp_m_mv(mv_s):

    for i in range(len(mv_s)):

        if mv_s[i]>10:

            m=10**((1e-3)*(0.3+1.87*mv_s+7.614*mv_s**2-1.698*mv_s**3+0.06096*mv_s**4))

        elif mv_s[i]<10:

            m=10**(0.477-0.135*mv_s+1.228*1e-2*mv_s**2-6.734*mv_s**3)

    return pd.DataFrame({'Mass':m,'Absolute Magnitude Mv':mv_s})
emp_m_mv(mv_sun)