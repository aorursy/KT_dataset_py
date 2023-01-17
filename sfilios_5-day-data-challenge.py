#import the necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

#read in the csv file
df = pd.read_csv('../input/museums.csv')
df.head()

df.describe()
#because I'm going to be graphing, I import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

#fill in the missing values in revenue
df['Filled_Revenue'] = df.Revenue.fillna(df.Revenue.mean())
#plot some charts
plt.scatter(df['Income'],df['Filled_Revenue'])
plt.title('Income versue Revenue')
plt.hist(df['Filled_Revenue'])
#I wanted a regression line in my scatter plot! So I am using seaborn.
import seaborn as sns
sns.regplot(df['Income'],df['Filled_Revenue'])
#I'm curious who has the highest income of all museums
df[df['Income']==df['Income'].max()]
#It looks like some institutions provide the total revenue for their institution and not per museum

#if the museum is reporting the same revenue for more than one institution, we could divide the revenue and income
#by the number of museums that it's split among
#I'll skip this for now

#I want to meet the challenge of performing a t-test
from scipy.stats import ttest_ind
#let's see if there is a difference between income from MA and CA located institutions
#first we select those museums that are in each state
MA = df['State (Physical Location)']=='MA'
CA = df['State (Physical Location)']=='CA'
compare = df[MA|CA]
mass = df[MA]
cal = df[CA]
mass.head()
#then we perform a t-test on the values
ttest_ind(a=mass.Filled_Revenue, b=cal.Filled_Revenue, nan_policy='omit', equal_var=False)
#there's no significant difference in revenue between museums that are in CA and MA.
plt.hist(mass.Filled_Revenue)
plt.hist(cal.Filled_Revenue)
#if we put the filled revenue for each in a similar plot, we see they don't look so different
sns.barplot(x=compare['State (Physical Location)'], y=compare['Filled_Revenue']).set_title('BOOM')
#you can see that CA located museums make more money, but not significantly more
#time to do a chi-square, as part of the data challenge. We should use two categorical variable.
#let's see if state has anything to do with museum type
import scipy.stats 
#pick the variables of interest
state = df['State (Physical Location)']
mus_type = df['Museum Type']

#use the pandas crosstab function to prepare a cross table so we can look at frequencies and expected freq.
x = pd.crosstab(mus_type, state)
#the crosstable gets passed to the chi square function

chi2, p, dif, expected = scipy.stats.chi2_contingency(x)
print (chi2, p)
# it seems like there's a relationship between the physical location and the type of museum
