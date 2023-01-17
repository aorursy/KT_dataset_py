import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings(action='ignore')
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity='all'
data= pd.read_csv('../input/Salaries.csv', index_col=[0])
data.shape
data.head()
data = data.convert_objects(convert_numeric=True)
data.info()
data.isna().sum()
data.dtypes
data.columns
#dropping unused columns

data.drop(['Notes', 'Status', 'Agency'], axis=1, inplace=True)
#Univariate Analysis

data[['BasePay', 'OvertimePay', 'OtherPay','Benefits', 'TotalPay', 'TotalPayBenefits']].describe()
data.hist(['BasePay', 'OvertimePay', 'OtherPay','Benefits', 'TotalPay', 'TotalPayBenefits'],grid=False, bins=30, figsize=(20,10))
for columns in ['BasePay', 'OvertimePay', 'OtherPay','Benefits', 'TotalPay', 'TotalPayBenefits']:

    plt.figure()

    data.boxplot([columns])   
data.isna().sum()
#replacing missing values with median

data['BasePay'].fillna(data['BasePay'].median(), inplace=True)
#cheching rows where TotalPayBenefits != TotalPay

data.loc[(data['TotalPay']!=data['TotalPayBenefits'])].head()
# BasePay+ OvertimePay+ OtherPay+ Benefits = TotalPayBenefits

# Benefits= TotalPayBenefits - TotalPay

#Dealing with missing values in TotalPayBenefits

data['Benefits']= data['TotalPayBenefits']- data['TotalPay']
#Top 10 profiles with highest median Total pay

hig_med_pay= data.groupby(by='JobTitle')['TotalPay'].agg(['count', 'median']).sort_values(by='median', ascending=False).head(10)

hig_med_pay

sns.barplot(x= hig_med_pay.index, y=hig_med_pay['median'].values)

plt.xticks(rotation=90)
#No of employees(data) for each year

data['Year'].value_counts()

sns.countplot(data['Year'])
#median value for Base Pay for Job title

hig_base_pay= data.groupby(by='JobTitle')['BasePay'].agg(['count', 'median']).nlargest(10,'median')

hig_base_pay

sns.barplot(x= hig_base_pay.index, y=hig_base_pay['median'].values)

plt.xticks(rotation=90)
#plotting heatmap

plt.figure(figsize=(10,8))

sns.heatmap(data.corr(), annot=True, cmap='viridis')
#Plot TotalPayBenefits vs Year

med_2011= data.loc[data['Year']==2011]['TotalPayBenefits'].median()

med_2012= data.loc[data['Year']==2012]['TotalPayBenefits'].median()

med_2013= data.loc[data['Year']==2013]['TotalPayBenefits'].median()

med_2014= data.loc[data['Year']==2014]['TotalPayBenefits'].median()



sns.barplot(x=['2011', '2012', '2013', '2014'], y=[med_2011, med_2012, med_2013, med_2014])

# the mediun pay for 2012, 2013, 2014 are amost equal
#Top no of jobs in Job title

data['JobTitle'].value_counts().nlargest(50)
plt.figure(figsize=(10,8))

sns.barplot(data['JobTitle'].value_counts().nlargest(50).index, data['JobTitle'].value_counts().nlargest(50).values, alpha=0.8)

plt.xticks(rotation=90)

plt.xlabel('Number of jobs', fontsize=16)

plt.ylabel("Job Title", fontsize=16)

plt.title("Number of jobs")

plt.show()
#Calculating the avg total pay values for top (max) Jobs in SF

a= data['JobTitle'].value_counts().head(10).index.tolist()

avg= []

for i in a:

    b= data.loc[data['JobTitle']==i]['TotalPay'].median()

    avg.append(b)



print(avg)    



sns.barplot(x=a, y=avg, palette='viridis', alpha=0.6)

plt.xticks(rotation=90)

plt.xlabel('Job Titles', fontsize=16)

plt.ylabel("Avg TotalPay", fontsize=16)

plt.title("Avg TotalPay for top 10 available jobs")

plt.show()
# Let’s find the employees who earn the highest totalPay amount in each year.

Years= [2011, 2012, 2013, 2014]

for i in Years:

    data.loc[(data['Year']==i)].sort_values(by='TotalPay', ascending=False).head(1)
Years= [2011, 2012, 2013, 2014]

for i in Years:

    a= data.loc[(data['Year']==i)].sort_values(by='TotalPay', ascending=False).head(1)

    print('In the year {} the highest earned person is {} with total income {}'.format(i, a['EmployeeName'].values[0], 

                                                                                       a['TotalPay'].values[0]))

    