# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
resource = pd.read_csv('../input/Resources.csv')
school = pd.read_csv('../input/Schools.csv')
donor = pd.read_csv('../input/Donors.csv')
donation = pd.read_csv('../input/Donations.csv')
teacher = pd.read_csv('../input/Teachers.csv')
project = pd.read_csv('../input/Projects.csv')
resource.info()
school.info()
donor.info()
donation.info()
teacher.info()
project.info()
donation_amount = donation.groupby('Project ID')['Donation Amount']
sns.distplot(np.log2(donation_amount.sum()))
plt.xlabel('log value of donation amount')
sns.distplot(np.log2(donation_amount.count()),kde=False)
plt.xlabel('donation count')
donation_donar = donation.groupby('Donor ID')['Donation Amount']
sns.distplot(np.log2(donation_donar.count()),kde=False)
plt.xlabel('donation count per donor')
donation_count_per_donor = donation_donar.count().sort_values(ascending=False).reset_index().rename(index=str, columns={'Donation Amount':'Donation Count'})
gt_100 = donation_count_per_donor[donation_count_per_donor['Donation Count']>100]
gt_100_dornor = pd.merge(gt_100,donor,how='left',on=['Donor ID'])
gt_100_dornor.head(5)
plt.figure(figsize=(10,12))
plt.title('distribution of state donor gt 100')
sns.countplot(y=gt_100_dornor['Donor State'])
plt.title('distribution of teacher')
sns.countplot(y=gt_100_dornor['Donor Is Teacher'])
prjoct_donation_count = donation.groupby('Project ID')['Donation ID'].count().sort_values(ascending=False).reset_index().rename(index=str, columns={'Donation ID':'Donation Count'})
gt_50_prjoct_donation_count = prjoct_donation_count[prjoct_donation_count['Donation Count']>=50]
gt_50_project = pd.merge(gt_50_prjoct_donation_count,project,how='left',on=['Project ID'])
sns.countplot(y=project['Project Resource Category'])
sns.countplot(y=gt_50_project['Project Resource Category'])
sns.countplot(y=project['Project Grade Level Category'])
sns.countplot(y=gt_50_project['Project Grade Level Category'])
gt_50_project.head(5)
fig,axes = plt.subplots(2,1,)
sns.distplot(np.log2(project['Project Cost'].dropna()),ax=axes[0])
sns.distplot(np.log2(gt_50_project['Project Cost'].dropna()),ax=axes[1])
