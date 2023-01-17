# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Luc d'Hauthuille [2017]

# Plots CO2 in ppm since 1958, then fits to find when we hit 450ppm. It

# uses data from the SCRIPPS institute.

# Source: [https://www.kaggle.com/ucsandiego/carbon-dioxide]

'''The carbon dioxide data was collected and published by the

University of California's Scripps Institution of Oceanography

under the supervision of Charles David Keeling with support from

the US Department of Energy, Earth Networks, and the National Science Foundation.'''



import numpy as np

import matplotlib.pyplot as plt

#Import the C02 from the csv file

csv = np.genfromtxt ('../input/archive.csv', delimiter=",")

year = csv[4:-11,2]

CO2 = csv[4:-11,5]



#Fit a polynomial, make years (x-points) for the polynomial

z = np.polyfit(year,CO2,3)

p = np.poly1d(z)

years_for_p = np.linspace(year[0], year[-1] + (year[-1] - year[0]), 2*len(year))



#Make a figure, add ticks...

fig = plt.figure()

ax = fig.add_subplot(111)

ax.yaxis.set_ticks([100, 150, 200, 250, 300, 350, 400, 450, 500])

ax.grid(True)



#Plot the measured data + fit

plt.plot(year,CO2, years_for_p,p(years_for_p), '-')

#Marker for 450ppm occur, which occurs ~ in the year 2035.2

plt.scatter(2035.2, 450, s=100, c='red', marker='o')

plt.legend( ('Data', 'Fit and extrapolate') )

plt.title('CO2 in ppm over time (Mauna Loa Observatory)')

plt.ylabel('Seasonally Adjusted CO2 (ppm)')

plt.show()