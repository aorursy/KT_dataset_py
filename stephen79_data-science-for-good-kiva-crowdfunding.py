# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
# display column limit

pd.set_option('display.max_columns',100)
import pandas as pd

kiva_loans = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")

kiva_mpi_region_locations = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")

loan_theme_ids = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")

loan_themes_by_region = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")
# look at the data

kiva_loans.head(3)
kiva_loans.shape
# exploratory data on kiva_loans

kiva_loans.isnull().sum()
kiva_loans.duplicated().sum()
kiva_loans.describe()
kiva_loans.dtypes
kiva_loans.shape
kiva_loans.head(2)
kiva_loans.drop(['loan_amount','use','country_code','currency','tags','posted_time','disbursed_time','funded_time','date'],axis=1,inplace=True)

kiva_loans.head()
# Distribution of funded_amount

sns.distplot(kiva_loans['funded_amount'])

plt.show()
# Most frequent activity used for loan

plt.figure(figsize=(15,8))

count = kiva_loans['activity'].value_counts().head(10)

sns.barplot(count.values, count.index, )

for i, v in enumerate(count.values):

    plt.text(0.8,i,v,color='k',fontsize=19)

plt.xlabel('Count',fontsize=12)

plt.ylabel('activity', fontsize=12)

plt.title('top 10 activity used for loans')
# Top sectors funded.

plt.figure(figsize=(15,8))

count = kiva_loans['sector'].value_counts().head(10)

sns.barplot(count.values, count.index,)

for i, v in enumerate(count.values):

    plt.text(0.8,i,v,color='k',fontsize=12)

plt.xlabel('count',fontsize=12)

plt.ylabel('sectors',fontsize=12)

plt.title('top 10 sectors funded', fontsize=12)

# Countries with the most borrowers.

plt.figure(figsize=(15,8))

count = kiva_loans['country'].value_counts().head(10)

sns.barplot(count.values, count.index,)

for i,v in enumerate (count.values):

    plt.text(0.8,i,v,color='k',fontsize=12)

plt.xlabel('count',fontsize=12)

plt.ylabel('country',fontsize=12)

plt.title('top 10 country with highest borrowers')
#Frequent regions where borrowers reside.

plt.figure(figsize=(15,8))

count = kiva_loans['region'].value_counts().head(10)

sns.barplot(count.values, count.index)

for i,v in enumerate(count.values):

    plt.text(0.8,i,v,color='k',fontsize=12)

plt.xlabel('count',fontsize=12)

plt.ylabel('regions',fontsize=12)

plt.title('top 10 regions with borrowers')
# The most frequent partner_ids.

plt.figure(figsize=(15,8))

count = kiva_loans['partner_id'].value_counts().head(10)

sns.barplot(count.values, count.index)

for i,v in enumerate (count.values):

    plt.text(0.8,i,v,color='k',fontsize=12)

plt.xlabel('count',fontsize=12)

plt.ylabel('partner_ids',fontsize=12)

plt.title('top 10 partner_ids')
# gender of the frequent borrowers.

kiva_loans['female_gender'] = kiva_loans['borrower_genders'].str.contains('female')

# create a gender column

kiva_loans.loc[kiva_loans['female_gender'] == True, 'gender'] = 'Female'

kiva_loans.loc[kiva_loans['female_gender'] == False, 'gender'] = 'Male'

# drop irrelevant columns

kiva_loans.drop(['borrower_genders', 'female_gender'], axis=1, inplace=True)
plt.figure(figsize=(15,8))

count = kiva_loans['gender'].value_counts()

sns.barplot(count.values,count.index,)

for i,v in enumerate (count.values):

    plt.text(0.8,i,v,color='k',fontsize=12)

plt.xlabel('Count', fontsize=12)

plt.ylabel('Gender',fontsize=12)

plt.title('gender of borrowers')
# Most method of repayment overall

plt.figure(figsize=(15,8))

count = kiva_loans['repayment_interval'].value_counts().head(10)

sns.barplot(count.values,count.index,)

for i,v in enumerate (count.values):

    plt.text(0.8,i,v,color='k',fontsize=12)

plt.xlabel('count',fontsize=12)

plt.ylabel('Repayment Interval',fontsize=12)

plt.title('Frequency of Repayment Methods',fontsize=12)
# Gender and sector

sns.countplot(y='sector',data=kiva_loans, hue='gender', order= kiva_loans['sector'].value_counts().index)
# gender and coutry 

count = sns.countplot(y='country',data=kiva_loans,hue='gender',order=kiva_loans['country'].value_counts().iloc[:10].index)
kiva_loans.head(3)
sns.countplot(y='repayment_interval',data=kiva_loans,hue='gender',order=kiva_loans['repayment_interval'].value_counts().index)
kiva_loans.groupby('sector')['repayment_interval'].value_counts()