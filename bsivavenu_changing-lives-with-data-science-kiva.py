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

datadir = '../input/'

loans = pd.read_csv(datadir+'kiva_loans.csv')
loans.head()
region = pd.read_csv(datadir+'loan_themes_by_region.csv')
region.head()
region_location = pd.read_csv(datadir+'kiva_mpi_region_locations.csv')
region_location.head()
loan_id = pd.read_csv(datadir+'loan_theme_ids.csv')
loan_id.head()
print(set(loans['country'])) #countries involved in kiva program
#countries with highest number of loans taken
loans.country.value_counts()[:5]

#less number of loans taken/sanctioned by/to these countries
loans.country.value_counts()[-5:]
#ascending order of sectors to which loans were sanctioned
loans.sector.value_counts()

import seaborn as sns
import matplotlib.pyplot as plt
loans.head(2)
# sns.barplot(x = 'sector',y='loan_amount',data=loans)
# plt.figure()
plt.figure(figsize=(10,5))
sns.barplot(x = 'sector',y='loan_amount',data = loans)
plt.xticks(rotation = 90)
plt.title('Loan amonut sanctioned to different sectors')
plt.show()
plt.figure(figsize=(10,5))
sns.countplot('sector',data=loans)
plt.xticks(rotation=90)
plt.title('different sectors and their count')
plt.show()
temp = pd.DataFrame(loans.country.value_counts().head(10))
temp.reset_index(inplace=True)
temp.columns = ['country','count']
temp
plt.figure(figsize=(10,5))
sns.barplot(x = 'country',y = 'count',data=temp)
plt.xticks(rotation=90)
plt.title('countries with more number of loans')
plt.show()
loans['repayment_interval'].value_counts().sort_values(ascending=False)
sns.countplot('repayment_interval',data=loans)
loans['activity'].value_counts().sort_values(ascending=False)[:5]
temp = pd.DataFrame(loans.activity.value_counts().head(10))
temp.reset_index(inplace=True)
temp.columns = ['activity','count']
temp
plt.figure(figsize=(10,5))
sns.barplot(x = 'activity',y = 'count',data=temp)
plt.xticks(rotation=90)
plt.show()
temp = pd.DataFrame(loans.region.value_counts().head(10))
temp.reset_index(inplace=True)
temp.columns = ['region','count']
print(temp)
sns.barplot(x = 'region',y = 'count',data=temp)
plt.xticks(rotation=90)
plt.show()
loans.term_in_months.value_counts().head(10)
loans[loans.lender_count==0].head()
#these are the loans with out invovling any lender
loans.lender_count.sort_values(ascending=False).head()
#dont get confused,the max lenders for a loan is 2986 and the left side number 70499 is the id to pull the record

max(loans.lender_count)

loans[loans.lender_count==2986]

