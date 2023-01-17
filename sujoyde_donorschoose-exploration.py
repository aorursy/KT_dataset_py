import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
donations = pd.read_csv('../input/Donations.csv')
donors = pd.read_csv('../input/Donors.csv', low_memory=False)
schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False)
teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines=False)
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False, parse_dates=["Project Posted Date","Project Fully Funded Date"])
resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False)
donations.info()
donors.info()
schools.info()
teachers.info()
projects.info()
resources.info()
donations.head()
# number of rows and columns in donations dataset
donations.shape
# unique number of project ids in donation dataset
donations['Project ID'].nunique()
# unique number of donation ids, donor ids in donation dataset
donations['Donation ID'].nunique(), donations['Donor ID'].nunique()
donations.describe()
# skewness of donation amount and donor cart sequence
donations['Donation Amount'].skew(), donations['Donor Cart Sequence'].skew()
#distribution of optional donation
donations['Donation Included Optional Donation'].value_counts(normalize = True)
# what is the average donation when the optional donation is no or yes
grouped = donations.groupby('Donation Included Optional Donation')['Donation Amount'].mean().reset_index()
grouped
donations['Donation Received Date'] = pd.to_datetime(donations['Donation Received Date'])
donations['Donation Received Date wd'] = donations['Donation Received Date'].dt.weekday_name
plt.rcParams['figure.figsize'] = [15, 4]
sns.barplot(x = 'Donation Received Date wd', y = 'Donation Amount', data = donations)
donations['Donation Received Date month'] = donations['Donation Received Date'].dt.month
plt.rcParams['figure.figsize'] = [15, 4]
sns.barplot(x = 'Donation Received Date month', y = 'Donation Amount', data = donations)
donations['Donation Received Date year'] = donations['Donation Received Date'].dt.year
sns.barplot(x = 'Donation Received Date year', y = 'Donation Amount', data = donations)
# lets check the counts of the year
donations['Donation Received Date year'].value_counts()
donors.head()
# how many cities of donor cities and states of donor dataset are there
donors['Donor City'].nunique(), donors['Donor State'].nunique()
# counts of whether a donor is a teacher or not
donors['Donor Is Teacher'].value_counts(normalize = True)
grouped = donations.groupby('Donor ID')['Donation Amount'].sum().reset_index()
donors = pd.merge(donors, grouped, on = 'Donor ID', how = 'inner')
donors.head(2)
# which state has the highest median donation amount
grouped = donors.groupby('Donor State')['Donation Amount'].median().reset_index()
grouped = grouped.sort_values(by = 'Donation Amount', ascending = False)
print (grouped.iloc[0])

# which state has the lowest median donation amount
print (grouped.iloc[-1])
donors[donors['Donor State'] == 'Idaho'].head(2)
# which city has the highest median donation amount
grouped = donors.groupby('Donor City')['Donation Amount'].median().reset_index()
grouped = grouped.sort_values(by = 'Donation Amount', ascending = False)
print (grouped.iloc[0])

# which city has the lowest median donation amount
print (grouped.iloc[-1])
# which state has the highest variation in donation amount
grouped = donors.groupby('Donor State')['Donation Amount'].std().reset_index()
grouped = grouped.sort_values(by = 'Donation Amount', ascending = False)
print (grouped.iloc[0])

# which state has the lowest variation in donation amount
print (grouped.iloc[-1])
# which state has the highest average donation amount
grouped = donors.groupby('Donor State')['Donation Amount'].mean().reset_index()
grouped = grouped.sort_values(by = 'Donation Amount', ascending = False)
print (grouped.iloc[0])
# which city has the highest variation in donation amount
grouped = donors.groupby('Donor City')['Donation Amount'].std().reset_index()
grouped = grouped.sort_values(by = 'Donation Amount', ascending = False)
print (grouped.iloc[0])

# which city has the lowest variation in donation amount
print (grouped.iloc[-1])
# how many average unique donors are there from a state
donors.shape[0]/donors['Donor State'].nunique()
# 90000 average donors are normally there from a state
donors[donors['Donor State'] == 'New York']['Donor ID'].nunique()
# plot distribution of donors from a state
grouped = donors.groupby('Donor State').size().reset_index()
grouped.columns = ['Donor State', 'Donor_count']
sns.distplot(grouped['Donor_count'])
grouped.describe()
# how many unique states are there?
grouped['Donor State'].nunique()
# how many states have donor count more than the average donor count
grouped[grouped['Donor_count']>grouped['Donor_count'].mean()].shape[0]
# which state has the highest number of donors?
print(grouped[grouped['Donor_count']==grouped['Donor_count'].max()])
# which state has the lowest number of donors?
print(grouped[grouped['Donor_count']==grouped['Donor_count'].min()])
