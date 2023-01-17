import pandas as pd 

import matplotlib.pyplot as plt 

import numpy as np 

from scipy.stats import norm 

import seaborn as sns 

plt.style.use('ggplot')

%matplotlib inline
data = pd.read_csv('../input/california-housing-prices/housing.csv')

data.head()
median_house_value = np.array(data.median_house_value)
plt.figure(figsize=(12, 8))

plt.title('Median House Value Distribution', size=18)

plt.xlabel('Value in $', size=18)

sns.distplot(median_house_value, fit=norm, color='blue', kde=False)
sample_num = 1000

sample_size = 30



mean_sample_values = []



for i in range(sample_num):

    sample_mean = np.mean(np.random.choice(median_house_value, sample_size, replace=True))

    mean_sample_values.append(sample_mean)   
plt.figure(figsize=(12, 8))

plt.title('Sample Mean of Median House Value Distribution', size=18)

plt.xlabel('Value in $', size=18)

sns.distplot(mean_sample_values, fit=norm, color='blue', kde=False)