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
df="../input/startup_funding.csv"
data=pd.read_csv(df)
print(data)
print(data.head())
print(data.columns)
print(data.info())
print("Identifying null values based on ascending order")
data.isnull().sum().sort_values(ascending =False)
data.drop(['Remarks'],axis=1,inplace=True)
print("After dropping Remarks column")
print(data.columns)
data['AmountInUSD'] = data['AmountInUSD'].apply(lambda x:float(str(x).replace(",","")))
print(data['AmountInUSD'].values)
data['AmountInUSD']=data['AmountInUSD'].astype(float)
print(data.info())
print(data['AmountInUSD'].values)
s=data['AmountInUSD'].mean()
data['AmountInUSD'].replace(np.NAN,s)
data['AmountInUSD'].sum()
data['Date']=data['Date'].replace({"12/05.2015":"12/05/2015"})
data['Date']=data['Date'].replace({"13/04.2015":"13/04/2015"})
data['Date']=data['Date'].replace({"15/01.2015":"15/01/2015"})
data['Date']=data['Date'].replace({"22/01//2015":"22/01/2015"})
print(data['Date'].values)
data["yearmonth"] = (pd.to_datetime(data['Date'],format='%d/%m/%Y').dt.year*100)+(pd.to_datetime(data['Date'],format='%d/%m/%Y').dt.month)
temp = data['yearmonth'].value_counts().sort_values(ascending = False)
print("Number of funding per month in decreasing order (Funding Wise)\n\n",temp)
year_month = data['yearmonth'].value_counts()
data['AmountInUSD'].min()
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import dateutil
import squarify
plt.figure(figsize=(20,15))
sns.barplot(year_month.index, year_month.values, alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('Year-Month of transaction', fontsize=15,color='red')
plt.ylabel('Number of fundings made', fontsize=15,color='red')
plt.title("Year-Month - Number of Funding Distribution", fontsize=18)
plt.show()
plt.figure(figsize=(15,8))
sns.barplot(data['yearmonth'],data['AmountInUSD'], alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('YearMonth', fontsize=12)
plt.ylabel('Amonut Of Investments', fontsize=12)
plt.title("YearMonth - Number of fundings distribution", fontsize=16)
plt.show()
print("Total number of startups")
len(data['StartupName'])
print("Unique startups")
len(data['StartupName'].unique())
tot = (data['StartupName'].value_counts())
c=0
for i in tot:
    if i > 1:
        c=c+1
print("Startups that got funding more than 1 times = ",c)
funt_count  = data['StartupName'].value_counts()
fund_count = funt_count.head(20)
print(fund_count)
plt.figure(figsize=(15,8))
sns.barplot(fund_count.index, fund_count.values, alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('Startups', fontsize=15)
plt.ylabel('Number of fundings made', fontsize=15)
plt.title("Startups-Number of fundings distribution", fontsize=16)
plt.show()
plt.figure(figsize=(15,8))
sns.barplot(fund_count.index, fund_count.values, alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('Startups', fontsize=15)
plt.ylabel('Number of fundings made', fontsize=15)
plt.title("Startups-Number of fundings distribution", fontsize=16)
plt.show()
print("Unique Industry verticals")
len(data['IndustryVertical'].unique())
IndustryVert = data['IndustryVertical'].value_counts().head(20)
print(IndustryVert)
plt.figure(figsize=(15,8))
sns.barplot(year_month.index, year_month.values, alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('Year-Month of transaction', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Year-Month - Number of Funding Distribution", fontsize=16)
plt.show()
