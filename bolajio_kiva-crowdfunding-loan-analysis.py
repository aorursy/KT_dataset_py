# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
kiva_loans= pd.read_csv("../input/kiva_loans.csv")
kiva_regions=pd.read_csv("../input/kiva_mpi_region_locations.csv")
kiva_theme=pd.read_csv("../input/loan_theme_ids.csv")
kiva_mpi=pd.read_csv("../input/loan_themes_by_region.csv")
kiva_loans.head(5)
kiva_loans.columns
kiva_regions.head(5)
kiva_theme.head(5)
kiva_mpi.head(10)
kiva_loans.describe(include = 'all')
#lets explore a breakdown of how the sectors that get loans
plt.figure(figsize=(13,8))
sectors = kiva_loans['sector'].value_counts()
sns.barplot(y=sectors.index, x=sectors.values, alpha=0.6)
plt.xlabel('Number of loans', fontsize=16)
plt.ylabel("Sectors", fontsize=16)
plt.title("Number of loans per sector")
plt.show();


#loan distribution
loan_dist = kiva_loans.groupby(['country'],as_index=False).mean()
fig,axa = plt.subplots(1,1,figsize=(15,6))
sns.distplot(loan_dist[loan_dist['funded_amount']<100000].loan_amount)
plt.show()

kiva_loans =kiva_loans[kiva_loans['funded_amount'] > 40000]
plt.figure(figsize=(30,15))
sns.countplot('country',data=kiva_loans)
plt.xticks(rotation=90)
plt.title('countries with $40,000 or more in loans')
plt.show()
#Let's explore repayment intervals
plt.figure(figsize=(15,8))
count = kiva_loans['repayment_interval'].value_counts().head(10)
sns.barplot(count.values, count.index, )
for i, v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=19)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Types of repayment interval', fontsize=12)
plt.title("Types of repayment intervals", fontsize=16)
#Let's explore Nigeria's data
Nigerian_loans_df = kiva_loans[(kiva_loans.country == 'Nigeria')]
Nigerian_loans_df.head(5)
Nigerian_loans_df.describe(include='all')
Philippines_loans_df = kiva_loans[(kiva_loans.country == 'Philippines')]
Philippines_loans_df.describe(include='all')

Kenyan_loans_df = kiva_loans[(kiva_loans.country == 'Kenya')]
Kenyan_loans_df.describe(include='all')
MPI = kiva_regions[['country','region', 'MPI']]
MPI.head(5)
#Let's explore mpi
merge_data= pd.merge(kiva_loans, MPI, how='left')
merge_data.count()
#MPI for countries greater than .4
kv = MPI[MPI['MPI'] > .40]
plt.figure(figsize=(50,25))
sns.barplot(y=kv.MPI, x=kv.country)
plt.xlabel('country', fontsize=16)
plt.ylabel('MPI')
plt.show()
#MPI compared with loan amount per country

MPIcomp=merge_data.groupby(['MPI','repayment_interval'])
data=MPIcomp

rows= merge_data['MPI'].unique()
columns=merge_data['repayment_interval'].unique()
plt.pcolor(data,cmap=plt.cm.Blues)

plt.show()