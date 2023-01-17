# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# our complete dataset
coffee_full = pd.read_csv('/kaggle/input/coffee_dataset.csv')
coffee_full.head()
np.random.seed(42)

#this is the only data you might actually get in the real world.
#lets build confidence interval based on the sample dataset 
coffee_sample = coffee_full.sample(200)
coffee_sample.head()
print(coffee_sample.shape)
display(coffee_sample.head())
#drink coffee
coffee_sample['drinks_coffee'].mean()
#dont drink coffee
1 - coffee_sample['drinks_coffee'].mean()
#average height of that individual who drink coffee 
filter_True = coffee_sample['drinks_coffee'] == True
sample_tmp = coffee_sample[filter_True]
avg_height = sample_tmp['height'].mean()
print(avg_height)
boots_mean = []
for i in range(10000):
    boots_sample = coffee_sample.sample(200, replace=True)
    sample_mean = boots_sample[boots_sample['drinks_coffee'] == False]['height'].mean()
    boots_mean.append(sample_mean)
    
#plot sample means 
plt.hist(boots_mean)
plt.show()
#since we dont have the population data we have to cut 2.5% from botton and top 

np.percentile(boots_mean, 2.5), np.percentile(boots_mean, 97.5)
#lets caclulate the non-coffee-drinker height from our actual data 
coffee_full[coffee_full['drinks_coffee']==False]['height'].mean()