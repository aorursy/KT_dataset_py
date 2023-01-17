import math

import seaborn as sns

import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

from scipy.stats import t as pt

import matplotlib.pyplot as plt

%matplotlib inline
# Read the data set

data = [(12.079,19.278),

(16.791,18.741),

(9.564,21.214),

(8.630,15.687),

(14.669,22.803),

(12.238,20.878),

(14.692,24.572),

(8.987,17.394),

(9.401,20.762),

(14.480,26.282),

(22.328,24.524),

(15.298,18.644),

(15.073,17.510),

(16.929,20.330),

(18.200,35.255),

(12.130,22.158),

(18.495,25.139),

(10.639,20.429),

(11.344,17.425),

(12.369,34.288),

(12.944,23.894),

(14.233,17.960),

(19.710,22.058),

(16.004,21.157),]



df = pd.DataFrame(data,columns = ['Congruent', 'Incongruent'])

df
#Find the mean,Median and standard deviation for congruent and incongruent

#Find the mean

c_mean = df['Congruent'].mean()

i_mean = df['Incongruent'].mean()



#Find the median

c_median = df['Congruent'].median()

i_median = df['Incongruent'].median()



#Find the standard deviation

c_std = df['Congruent'].std()

i_std = df['Incongruent'].std()



#print mean,median and standard deviation in a table

ls = [[int(24),int(24)],[c_mean,i_mean],[c_median,i_median],[c_std,i_std]]

detail = pd.DataFrame(ls,index=['Sample Size','Mean','Median','Standard Deviation'],columns=['Congruent','Incongruent'])

detail
#More Detail About the dataset

df.describe()
#Make a boxplot for both datasets

sns.set_style("whitegrid")

sns.boxplot(data=df[['Congruent', 'Incongruent']], orient="v",width=0.4, palette="colorblind");

plt.ylabel("Time");
#Plot a Graph for congruent dataset

sns.distplot(df['Congruent'])

plt.xlabel("Time");

plt.ylabel("Frequency");

plt.title("Response Time for Congruent Words");
#Plot a Graph for congruent dataset

sns.distplot(df['Incongruent'])

plt.xlabel("Time");

plt.ylabel("Frequency");

plt.title("Response Time for Incongruent Words");

plt.show()
#Compare Both the datasets and make a graph

sns.distplot(df['Congruent'],label = "Congruent")

sns.distplot(df['Incongruent'],label = "Incongruent")

plt.xlabel("Time");

plt.ylabel("Frequency");

plt.title("Response Time For Congruent Vs Incongruent Words");

plt.legend();
#find t-critical value for 95% confidance interval and 23 degree of freedom for two tailed test



print("t-critical value for two tailed test is: ",round(pt.ppf(0.975,23),4))
#find the diffenence of each data

df['difference'] = df['Congruent'] - df['Incongruent']

df
#sd and mean of the differenced dataset

s_std = df['difference'].std()

print("Standard Deviation of the diffenenced dataset: ", round(s_std,4))

s_mean = c_mean - i_mean

print("Mean of Difference: ", round(s_mean,4))
# Calculate the t-value

t_value = s_mean/(s_std/math.sqrt(24))

print("t-Value is: ",t_value)