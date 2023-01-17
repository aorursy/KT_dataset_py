# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  #for plotting graphs
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For exadmple, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

salaries_by_region = pd.read_csv('../input/salaries-by-region.csv')
salaries_by_college_type = pd.read_csv('../input/salaries-by-college-type.csv')
degrees_that_pay_back = pd.read_csv('../input/degrees-that-pay-back.csv')


# Any results you write to the current directory are saved as output.
print ('Shape: ',salaries_by_region.shape,'\nTotal cells: ',np.product(salaries_by_region.shape))
print ('Empty cells columns wise: ',salaries_by_region.isnull().sum())
#Drops columns which has at least one missing value
salaries_by_region = salaries_by_region.dropna(axis=1)
print (salaries_by_region.shape)

#We can rename some columns names for easier use
salaries_by_region.rename(columns = {'Starting Median Salary':'SMS','Mid-Career Median Salary':'Mid_MS',
                                     'Mid-Career 25th Percentile Salary':'Mid_25MS',
                                     'Mid-Career 75th Percentile Salary':'Mid_75MS'},inplace=True)

#When you run this twicw, it will show error as it would hav already been replaced
column_containing_dollar_symbols = ['SMS','Mid_MS','Mid_25MS','Mid_75MS']
for i in column_containing_dollar_symbols:
    salaries_by_region[i] = salaries_by_region[i].str.replace('$','')
    salaries_by_region[i] = salaries_by_region[i].str.replace(',','')
    salaries_by_region[i] = pd.to_numeric(salaries_by_region[i])
#Pre processing the same way like salaries by region.
print ('Salaries by college type shape: ',salaries_by_college_type.shape,'\nTotal cells: ',np.product(salaries_by_college_type.shape))

salaries_by_college_type.rename(columns = {'Starting Median Salary':'SMS','Mid-Career Median Salary':'Mid_MS',
                                     'Mid-Career 25th Percentile Salary':'Mid_25MS',
                                     'Mid-Career 75th Percentile Salary':'Mid_75MS'},inplace=True)

salaries_by_college_type.dropna(axis=1,inplace=True)

print ('Empty cells columns wise in salaries by region type \n',salaries_by_college_type.isnull().sum())

for i in column_containing_dollar_symbols:
    salaries_by_college_type[i] = salaries_by_college_type[i].str.replace('$','')
    salaries_by_college_type[i] = salaries_by_college_type[i].str.replace(',','')
    salaries_by_college_type[i] = pd.to_numeric(salaries_by_region[i])

print ('Degree that pay back shape: ',degrees_that_pay_back.shape,'\nTotal cells: ',np.product(degrees_that_pay_back.shape))

degrees_that_pay_back.rename(columns = {'Undergraduate Major':'UM', 'Starting Median Salary':'SMS', 'Mid-Career Median Salary':'Mid_MS', 
                                           'Percent change from Starting to Mid-Career Salary':'Change', 'Mid-Career 10th Percentile Salary':'M10', 
                                           'Mid-Career 25th Percentile Salary':'M25', 'Mid-Career 75th Percentile Salary':'M75', 
                                           'Mid-Career 90th Percentile Salary':'M90'},inplace=True)
degrees_that_pay_back.dropna(axis=1,inplace=True)

print ('Empty cells columns wise in degrees that pay back \n',degrees_that_pay_back.isnull().sum())

for i in column_containing_dollar_symbols:
    degrees_that_pay_back[i] = degrees_that_pay_back[i].str.replace('$','')
    degrees_that_pay_back[i] = degrees_that_pay_back[i].str.replace(',','')
    degrees_that_pay_back[i] = pd.to_numeric(salaries_by_region[i])

df = salaries_by_region.groupby(['Region']).mean()

df.rename(index={'Northeastern':'NE','California':'CF','Southern':'S','Midwestern':'MW','Western':'W'},inplace=True)

df['SMS'] = df['SMS']/1000
df['Mid_MS'] = df['Mid_MS']/1000
df['Mid_25MS'] = df['Mid_25MS']/1000
df['Mid_75MS'] = df['Mid_75MS']/1000

print ('Plot y values are in thousand dollars')
fig , ax = plt.subplots(1,4)
fig.set_size_inches(18.5, 10.5)

df.sort_values(by='SMS',inplace=True,ascending=False)
sns.barplot(x = df.index.values,y='SMS',data=df,ax = ax[0])
ax[0].set_title('Starting Median Salary')

df.sort_values(by='Mid_MS',inplace=True,ascending=False)
sns.barplot(x = df.index.values,y='Mid_MS',data=df,ax=ax[1])
ax[1].set_title('Mid Median Salary')

df.sort_values(by='Mid_25MS',inplace=True,ascending=False)
sns.barplot(x = df.index.values,y='Mid_25MS',data=df,ax=ax[2])
ax[2].set_title('Mid 25 Percentile Median Salary')

df.sort_values(by='Mid_75MS',inplace=True,ascending=False)
sns.barplot(x = df.index.values,y='Mid_75MS',data=df,ax=ax[3])
ax[3].set_title('Mid 75 Percentile Median Salary')

plt.show()


df = salaries_by_college_type.groupby(['School Type']).mean()

df.rename(index = {'Engineering':'E','Party':'P','State':'S','Liberal Arts':'LA','Ivy League':'IL'},inplace=True)

df['SMS'] = df['SMS']/1000
df['Mid_MS'] = df['Mid_MS']/1000
df['Mid_25MS'] = df['Mid_25MS']/1000
df['Mid_75MS'] = df['Mid_75MS']/1000

print ('Plot y values are in thousand dollars')
fig , ax = plt.subplots(1,4)
fig.set_size_inches(18.5, 10.5)

df.sort_values(by='SMS',inplace=True,ascending=False)
sns.barplot(x = df.index.values,y='SMS',data=df,ax = ax[0])
ax[0].set_title('Starting Median Salary')

df.sort_values(by='Mid_MS',inplace=True,ascending=False)
sns.barplot(x = df.index.values,y='Mid_MS',data=df,ax=ax[1])
ax[1].set_title('Mid Median Salary')

df.sort_values(by='Mid_25MS',inplace=True,ascending=False)
sns.barplot(x = df.index.values,y='Mid_25MS',data=df,ax=ax[2])
ax[2].set_title('Mid 25 Percentile Median Salary')

df.sort_values(by='Mid_75MS',inplace=True,ascending=False)
sns.barplot(x = df.index.values,y='Mid_75MS',data=df,ax=ax[3])
ax[3].set_title('Mid 75 Percentile Median Salary')

plt.show()

degrees_that_pay_back.rename(columns={'Undergraduate Major':'UM',
                                      'Mid-Career 10th Percentile Salary':'M_10', 'Mid-Career 90th Percentile Salary':'M90'},inplace=True)
by_SMS = degrees_that_pay_back.sort_values(by='SMS',ascending=False)[0:3]['UM']
by_MC_MS = degrees_that_pay_back.sort_values(by='Mid_MS',ascending=False)[0:3]['UM']
by_M90 = degrees_that_pay_back.sort_values(by='M90',ascending=False)[0:3]['UM']

degree = list(by_SMS) + list(by_MC_MS) + list(by_M90)

degree = set(degree) #This removes duplicate elements in the list

df = degrees_that_pay_back.loc[degrees_that_pay_back['UM'].isin(degree)]
df = df.drop(['M10','M25','M75'],axis = 1)

df.sort_values(by='Change',inplace=True,ascending=False)
fig,ax = plt.subplots(1,1)
sns.barplot(x = 'UM',y='Change',data=df,ax = ax)
plt.xticks(rotation = 90)
plt.show()

df = df.drop(['Change'],axis = 1)

df['M90'] = df['M90'].str.replace('$','')
df['M90'] = df['M90'].str.replace(',','')
df['M90'] = pd.to_numeric(df['M90'])

df.drop(['Change'],axis=1,inplace=True)
plt.plot( 'UM', 'SMS', data=df, marker='o', markerfacecolor='skyblue', markersize=6, color='skyblue', linewidth=3)
plt.plot( 'UM', 'Mid_MS', data=df, marker='o', markerfacecolor='green', markersize=6, color='green', linewidth=3)
plt.plot( 'UM', 'M90', data=df, marker='o', markerfacecolor='red', markersize=6, color='red', linewidth=3)
plt.xticks(rotation=90)
plt.legend()
plt.show()