# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
from IPython import display
#collection of machine learning algorithms
import sklearn
import time

#charting tools
import matplotlib.pyplot as plt
import seaborn as sns

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
## Load Data
df_donations = pd.read_csv('../input/Donations.csv')
print(df_donations.columns)
print('-'*25)
df_donors = pd.read_csv('../input/Donors.csv')
print(df_donors.columns)
#Donations and Donors can be merged on Donor ID.
print('-'*50)
df_projects = pd.read_csv('../input/Projects.csv')
#project Data looks strange. 
df_resources = pd.read_csv('../input/Resources.csv')
print(df_resources.columns)
print('-'*25)
df_schools = pd.read_csv('../input/Schools.csv')
print(df_schools.columns)
print('-'*25)
df_teachers = pd.read_csv('../input/Teachers.csv')
print(df_teachers.columns)
print('-'*25)
#merge two dataset on Donor ID.
set1 = pd.merge(df_donations, df_donors, on = 'Donor ID')
print(set1.info())
#print missing data
print(pd.isnull(set1).sum())

print('Min amount is: $', round(set1['Donation Amount'].min(),2))
print('Max amount is: $', round(set1['Donation Amount'].max(),2))
print('Median amount is: $', round(set1['Donation Amount'].median(),2))
print('Mean amount is: $', round(set1['Donation Amount'].mean(),2))
chart1 = set1[df_donations['Donation Amount']>0]['Donation Amount']
sns.distplot(chart1, bins=40, kde=False);
chart2 = set1[df_donations['Donation Amount']<100]['Donation Amount']
sns.distplot(chart2, bins=40, kde=False);
set1['Donor ID'].value_counts()[0:10]
set1['Donation Amount'].groupby(df_donations['Donor ID']).sum().sort_values(ascending=False)[0:10]
by_freq = set1['Donor ID'].value_counts().to_dict()
#dictionary of frequency data
by_value = set1['Donation Amount'].groupby(df_donations['Donor ID']).sum().to_dict()
#dictionary of sum of donation amount
#then build two columns "Frequency", "Total Amount", and "Average Amount"
set1['Frequency'] = set1['Donor ID'].map(by_freq)
set1['Total Amount'] = set1['Donor ID'].map(by_value)
set1['Average Amount'] = set1['Total Amount'] / set1['Frequency']
#Top 10 Donors in Frequency, Total Amount, and Average Amount
#need to re-write code that prints top 10 Donor ID's by Frequency, and Total Amount. 
#The table needs to have Donor ID, Frequency, Total Amount, and Average Amount

tmp = set1['Donor State'].value_counts()
df1 = pd.DataFrame({'State': tmp.index,'Number of donations': tmp.values})
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,15))
s = sns.barplot(ax = ax1, x = 'Number of donations', y="State",data=df1)
tmp = set1.groupby('Donor State')['Donation Amount'].sum().sort_values(ascending = False)
df1 = pd.DataFrame({'State': tmp.index,'Total sum of donations in 10M': tmp.values})
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,15))
s = sns.barplot(ax = ax1, x = 'Total sum of donations in 10M', y="State",data=df1)



