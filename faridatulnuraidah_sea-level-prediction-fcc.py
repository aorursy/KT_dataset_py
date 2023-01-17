# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import linregress
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/sealeveldatasets/epa-sea-level.csv")
df
sr1 = pd.Series([int(i) for i in range(1880, 2051)])
recent = df[df['Year']>=2000]
yearto2050_2 = pd.Series([int(i) for i in range(2000, 2051)])
recent['Year'].append(yearto2050_2,ignore_index=True)
recent
# * Use matplotlib to create a scatter plot using the "Year" column as the x-axis and 
# the "CSIRO Adjusted Sea Level" column as the y-axix.
df.plot.scatter('Year','CSIRO Adjusted Sea Level',label='original data',figsize=(12, 7))
# * Use the `linregress` function from `scipi.stats` to get the slope and y-intercept of the line of best fit. 
# Plot the line of best fit over the top of the scatter plot. Make the line go through the year 2050 
# to predict the sea level rise in 2050.
yearto2050 = pd.Series([int(i) for i in range(1880, 2051)])
slope, intercept, r_value, p_value, std_err = stats.linregress(df['Year'],df['CSIRO Adjusted Sea Level'])
plt.plot(yearto2050, intercept + slope*yearto2050, 'r', label='best fit')

# * Plot a new line of best fit just using the data from year 2000 through the most recent year in the dataset. 
# Make the line also go through the year 2050 to predict the sea level rise in 2050 if the rate of rise continues 
# as it has since the year 2000.
recent = df[df['Year']>=2000]
slope_y, intercept_y, r_y, p_y, std_err_y = stats.linregress(recent['Year'],recent['CSIRO Adjusted Sea Level'])

yearto2050_2 = pd.Series([int(i) for i in range(2000, 2051)])
# recent.append(yearto2050_2,ignore_index=True)

plt.plot(yearto2050_2, intercept_y + slope_y*yearto2050_2, 'g', label='>year 2000')
# * The x label should be "Year", the y label should be "Sea Level (inches)", 
# and the title should be "Rise in Sea Level".

plt.xlabel("Year")
plt.ylabel("Sea Level (inches)")
plt.title("Rise in Sea Level")
plt.legend(fontsize="medium")
plt.show()