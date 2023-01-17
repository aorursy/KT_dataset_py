# import important libraries



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('../input/500-person-gender-height-weight-bodymassindex/500_Person_Gender_Height_Weight_Index.csv')

data.head()
# Let's get weight data first by dropping height column

weight_data = data.drop(['Height'], axis=1)

weight_data
weight = weight_data['Weight']



# Sample mean

print(f'Sample Mean = {np.mean(weight)}')
# Sample median

print(f'Sample Median = {np.median(weight.sort_values(ascending = True))}')
# Sample mode



# First, let's find the highest frequency

highest_freq = max(weight.value_counts())



# Then we print all weight values that have same frequency with highest_freq

print(weight.value_counts()[weight.value_counts() == highest_freq])
# Range



print(f'Range = {max(weight) - min(weight)}')
# Sample Variance



avg = np.mean(weight)

sigma = sum([(weight[i] - avg)**2 for i in range(len(weight))])

variance = sigma / (len(weight) - 1)

print('Variance =',variance)
# Sample standard deviation



print('Sample Standard Deviation =',variance**0.5)
# Let's sort the data first

weight = weight.sort_values(ascending = True).reset_index(drop = True)

print(weight)
# 15th percentile



n = len(weight)

idx = n * 0.15 # idx = 75.0

p15 = (weight[idx] + weight[idx + 1]) / 2

print('15th percentile =',p15)
# 90th percentile



idx = n * 0.9 # idx = 450.0

p90 = (weight[idx] + weight[idx + 1]) / 2

print('90th percentile =',p90)
# IQR is a difference between Q3 and Q1



idx = n * 0.25 # 125.0

q1 = (weight[idx] + weight[idx + 1]) / 2



idx = n * 0.75 # 375.0

q3 = (weight[idx] + weight[idx + 1]) / 2



IQR = q3 - q1

print('IQR =',IQR)
# RLB



RLB = q1 - 1.5*IQR

print('RLB =',RLB)



# RUB



RUB = q3 + 1.5*IQR

print('RUB =',RUB)
total_outliers = len(weight[weight > RUB]) + len(weight[weight < RLB])

print(f'We have {total_outliers} outliers in our data')
sns.set_style('darkgrid')



f, ax = plt.subplots(figsize = (10,8))

ax = sns.boxplot(data = weight_data, y = weight_data['Weight'], x = weight_data['Gender'], palette = 'pastel')

ax.set_title(label = 'Weight by Gender', size = 20)

plt.ylabel('Weight', size = 15)

plt.xlabel('Gender', size = 15)

plt.show()



f, ax = plt.subplots(figsize = (15,8))

ax = sns.boxplot(data = weight_data, y = weight_data['Weight'], x = weight_data['Index'], hue = 'Gender' ,palette = 'pastel')

ax.set_title(label = 'Weight by Index', size = 20)

plt.ylabel('Weight', size = 15)

plt.xlabel('Index', size = 15)

plt.show()
n_group = (max(weight) - min(weight)) // 10



f, ax = plt.subplots(1,1, figsize=(8,4))

ax = sns.distplot(weight_data['Weight'], bins = n_group, color = '#156DF3', kde = False)

plt.ylabel('Frequency', size = 15)

plt.xlabel('Weight', size = 15)

plt.title('Weight Histogram', size = 20)



plt.show()