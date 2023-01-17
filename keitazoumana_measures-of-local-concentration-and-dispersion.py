import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/forestfires/forestfires.csv')

data.head()
from statistics import mean 

from scipy import stats 
mean_value = data['temp'].mean()

mode_value = data['temp'].mode()

median_value = np.median(data['temp'])  



print('Mean : {}'.format(mean_value))

print('Mode : {}'.format(mode_value))

print('Median : {}'.format(median_value))
from statistics import stdev
std_value = data['temp'].std()

var_value = data['temp'].var()

print('Standard deviation: {}'.format(std_value))

print('Variance : {}'.format(var_value))
from scipy.stats import skew
skew_value = data['temp'].skew()

kurt_value = data['temp'].kurt()



print('Skeewness: {}'.format(skew_value))

print('Kurtosis: {}'.format(kurt_value))
data.describe()
sns.set(color_codes=True)

sns.kdeplot(data['temp'], shade=True)



# Mean and median values. 

plt.axvline(data['temp'].mean(), 0, 1, color = 'b') # shows the mean value

plt.axvline(data['temp'].median(), 0, 1, color = 'g') # shows the median value

#plt.xlabel()

#plt.ylabel()
description = data.describe()

description['temp']
sns.set(color_codes=True)

sns.kdeplot(data['temp'], shade=True)



plt.axvline(description['temp']['25%'], 0, 1, color='g') # First percentile

plt.axvline(description['temp']['75%'], 0, 1, color='g') # Third percentile



IQR = description['temp']['75%'] - description['temp']['25%']



# Show the lower and upper outlier limits 

lower_outliers = description['temp']['75%'] - 1.5 * IQR

upper_outliers = description['temp']['25%'] + 1.5 * IQR 



plt.axvline(lower_outliers, 0, 1, color='r')

plt.axvline(upper_outliers, 0, 1, color='r')