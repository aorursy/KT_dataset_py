# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
pd.set_option('display.float_format', lambda x: '%.3f' % x)
%matplotlib inline

# Any results you write to the current directory are saved as output.
Donors = pd.read_csv("../input/Donors.csv")
Donations = pd.read_csv("../input/Donations.csv")
Resources = pd.read_csv("../input/Resources.csv")
Teachers = pd.read_csv("../input/Teachers.csv")
Projects = pd.read_csv("../input/Projects.csv")
Schools = pd.read_csv("../input/Schools.csv")
Projects.head()
def to_str_currency(amount):
    return '{:20,.2f}'.format(amount).strip()

fund_demanded = np.sum(Projects['Project Cost'])
donation_received = np.sum(Donations['Donation Amount'])

print('Requested amount %s.' % (to_str_currency(fund_demanded)))
print('Raised amount %s' %  to_str_currency(donation_received))
print('Short by %s' %to_str_currency(fund_demanded - donation_received))
Donations['year'] = pd.DatetimeIndex(Donations['Donation Received Date']).year
Donations['month'] = pd.DatetimeIndex(Donations['Donation Received Date']).month

Donation_Group_by_Month = Donations.groupby(by='month').sum()
total = np.sum(Donation_Group_by_Month['Donation Amount'])
Donation_Group_by_Month['donation_by_perct'] = (Donation_Group_by_Month['Donation Amount']*100)/total
Donation_Group_by_Month['month'] = Donation_Group_by_Month.index
sns.barplot(x="month", y="donation_by_perct", data=Donation_Group_by_Month)
Donation_by_y_m = Donations[['Donation Amount', 'year', 'month']].groupby(by = ['year', 'month'])
Donations[['Donation Amount', 'year', 'month']].head(1)
copy_of_donation = Donations[['Donation Amount', 'month', 'year']].copy(deep=True)
# Remove outlier.
expectionally_donated_amount = 15600
copy_of_donation_without_outlier = copy_of_donation[copy_of_donation['Donation Amount'] != expectionally_donated_amount]

a4_dims = (17.7, 6.27)
fig, ax = plt.subplots(figsize=a4_dims)

sns.barplot(x="year", y="Donation Amount", hue="month", data=copy_of_donation_without_outlier, ax = ax)
fig2, ax2 = plt.subplots(figsize=a4_dims)

sns.barplot(x="month", y="Donation Amount", hue="year", data=copy_of_donation_without_outlier, ax = ax2)
Donations.head(2)