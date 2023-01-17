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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
## Load our data into a dataframe
bf_data=pd.read_csv("../input/BlackFriday.csv")
## Let us have a look at some data points
bf_data.head()
## Data info
bf_data.info()
## Let us check some summary stats for our numerical variables
bf_data.describe()
## Let us have a closer look at the purchase variable to understand it better.
plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
sns.distplot(bf_data.Purchase)
plt.subplot(1,2,2)
sns.distplot(np.log(bf_data.Purchase))
plt.xlabel('Log of Purchase')
plt.tight_layout()
## Lets first check the how many F and M we have in our data
bf_gender=bf_data[['User_ID','Gender']].drop_duplicates()
bf_gender['Gender'].value_counts()
bf_data_grped=bf_data[['User_ID','Gender','Purchase']].groupby(['User_ID','Gender'],as_index=False).sum().rename(columns={'Purchase':'Total Purchase'})
plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
sns.boxplot(x='Gender',y='Total Purchase',data=bf_data_grped)
plt.subplot(1,2,2)
sns.barplot(x='Total Purchase',y='Gender',data=bf_data_grped)
plt.xlabel('M vs F distribution of Total Purchase')
plt.legend(('Male','Female'),loc='upper right')
plt.show()
bf_age_grped=bf_data[['User_ID','Age','Purchase']].groupby(['User_ID','Age'],as_index=False).sum().rename(columns={'Purchase':'Total Purchase'})
plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
sns.boxplot(x='Age',y='Total Purchase',data=bf_age_grped)
def age_group_new(age_grp):
    if age_grp=='0-17':
        return 'Youngsters'
    elif age_grp in ('18-25','26-35'):
        return 'Millenials'
    else :
        return 'Non-Millenials'
bf_age_grped['New_age_grp']=bf_age_grped.Age.map(age_group_new)
plt.subplot(1,2,2)
sns.boxplot(x='New_age_grp',y='Total Purchase',data=bf_age_grped)
plt.xlabel('Custom Age groups')
plt.show()
print('Counts of age groups')
print(bf_age_grped.New_age_grp.value_counts())
print('---------------------------------------')
print('Median purchase by age group')
print(bf_age_grped[['Total Purchase','New_age_grp']].groupby(['New_age_grp']).median())
print('---------------------------------------')
print('Mean purchase by age group')
print(bf_age_grped[['Total Purchase','New_age_grp']].groupby(['New_age_grp']).mean())
plt.figure(figsize=(20,5))
sns.barplot(x='Age',y='Total Purchase',data=bf_age_grped)
plt.xlabel('Age groups')
plt.show()
print('Counts of age groups')
print(bf_age_grped.Age.value_counts())
print('---------------------------------------')
print('Median purchase by age group')
print(bf_age_grped[['Total Purchase','Age']].groupby(['Age']).median())
print('---------------------------------------')
print('Mean purchase by age group')
print(bf_age_grped[['Total Purchase','Age']].groupby(['Age']).mean())
bf_age_grped=bf_data[['User_ID','Gender','Age','Purchase']].groupby(['User_ID','Gender','Age'],as_index=False).sum().rename(columns={'Purchase':'Total Purchase'})
plt.figure(figsize=(20,5))
sns.barplot(x='Age',y='Total Purchase',hue='Gender',data=bf_age_grped,palette="Set1")
plt.show()
plt.figure(figsize=(20,5))
sns.boxplot(x='Age',y='Total Purchase',hue='Gender',data=bf_age_grped,palette="Set1")
plt.show()
bf_occp_grped=bf_data[['User_ID','Occupation','Purchase']].groupby(['User_ID','Occupation'],as_index=False).sum().rename(columns={'Purchase':'Total Purchase'})
bf_occp_grped['Occupation']=bf_occp_grped['Occupation'].astype('object')
## Let's check whats the percentage of occupations in our data first !!
plt.figure(figsize=(20,10))
bf_occp_grped.Occupation.value_counts().plot.pie(autopct='%1.1f%%')
plt.show()
## Let's check the total purchase value by occupation and plot it
bf_occp_grped_sum=bf_occp_grped[['Occupation','Total Purchase']].groupby(['Occupation'],as_index=False).sum()
ind=np.arange(len(bf_occp_grped_sum['Occupation']))
plt.figure(figsize=(20,10))
plt.barh(ind,bf_occp_grped_sum['Total Purchase'])
plt.yticks(ind,bf_occp_grped_sum['Occupation'])
plt.xlabel('Total Purchase')
plt.ylabel('Occupation')
plt.title('Total Purchase by Occupation')
#sns.barplot(y='Total Purchase',x='Occupation',data=bf_occp_grped,ci=None)
plt.show()
bf_stay_grped=bf_data[['User_ID','Stay_In_Current_City_Years','Purchase']].groupby(['User_ID','Stay_In_Current_City_Years'],as_index=False).sum().rename(columns={'Purchase':'Total Purchase'})
## Let's first check the frequency of different years of stay
plt.figure(figsize=(20,5))
sns.countplot(x='Stay_In_Current_City_Years',data=bf_stay_grped,order=['0','1','2','3','4+'])
plt.xlabel('Stay in current city in years')
plt.show()
## Let's check how purchase amount differs by years of stay.
plt.figure(figsize=(20,5))
sns.barplot(x='Stay_In_Current_City_Years',y='Total Purchase',data=bf_stay_grped,order=['0','1','2','3','4+'])
plt.show()
plt.figure(figsize=(20,10))
sns.boxplot(x='Stay_In_Current_City_Years',y='Total Purchase',data=bf_stay_grped,order=['0','1','2','3','4+'])
plt.show()
