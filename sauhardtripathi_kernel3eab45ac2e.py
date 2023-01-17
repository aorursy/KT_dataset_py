# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing the data
df=pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
#checking the head of the data
df.head(10)
#Renaimg the columns.
df.rename(columns={" gdp_for_year ($) ":
                  "gdp_for_year", "gdp_per_capita ($)":
                  "gdp_per_capita"},inplace=True)
#shows the number and types of object.
df.info()
#to check null value in the data.
df.isnull().sum()
#pairplot using seaborn
import seaborn as sns
sns.pairplot(df)
#correlation matrix
df.corr()
#heatmap of the data
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True,cmap='viridis')
plt.tight_layout()
df.groupby('year').sum()['suicides_no'].plot(figsize=(10,6))
plt.ylabel('suicides_no')
plt.xlabel('year')
plt.figure(figsize=(12,6))
data=df.groupby(['year','sex']).agg('mean').reset_index()
sns.lineplot(x='year',y='suicides/100k pop',data=data,hue='sex')
plt.xlabel('year')
plt.ylabel('suicides/100k pop')
plt.title('MEAN SUICIDES PER 100K POPULATION OF MALE AND FEMALE ')
plt.figure(figsize=(12,6))
sns.barplot(x='year',y='suicides_no',data=df,hue='sex')
plt.tight_layout()
plt.title('Suicide count on the basis of year and sex')
#yearwise suicide count
plt.figure(figsize=(12,6))
sns.barplot(x='year',y='suicides/100k pop',data=df,hue='sex')
plt.tight_layout()
plt.title('Yearwise Suicide count per 100k population ')
df['age'].value_counts()
df['age'].unique()
df.groupby('age').sum()['suicides_no'].reset_index().sort_values(by='suicides_no',ascending=False)
#predifing the orders.
sorder=['5-14 years','15-24 years','25-34 years', '35-54 years', 
       '55-74 years','75+ years']
#age-wise suicide count
plt.figure(figsize=(12,6))
sns.barplot(x='age',y='suicides_no',data=df,order=sorder)
plt.tight_layout()
plt.title('Age-wise suicide count ')
#age-wise suicide count 
plt.figure(figsize=(12,6))
sns.barplot(x='age',y='suicides_no',data=df,order=sorder,hue='sex')
plt.tight_layout()
plt.title('Age-wise suicide count based on sex')
plt.figure(figsize=(10,6))
data=df.groupby(['year','sex','age']).agg('mean').reset_index()
sns.relplot(x='year',y='suicides/100k pop',data=data,hue='sex',col='age',col_wrap=3,kind='line')
plt.suptitle('EVOLUTION OF SUICIDE BY SEX AND AGE')
plt.subplots_adjust(top = 0.9)
#Country-wise suicide count.
df[(df['year'] == 1985)].groupby('country')[['suicides_no']].sum().reset_index().sort_values(by='suicides_no',ascending=False)
df[df['year']==2016].groupby('country')[['suicides_no']].sum().reset_index().sort_values(by='suicides_no',ascending=False)
df.groupby('year').sum()['suicides_no'].reset_index().sort_values(by='suicides_no',ascending=False)
df.groupby(['country','year']).sum()['suicides/100k pop'].reset_index().sort_values(by='suicides/100k pop',ascending=False).head(20)
plt.figure(figsize=(10,6))
sns.barplot(x='generation',y='suicides_no',data=df,hue='sex',palette='Set1')
plt.title('Generation-wise suicide count of males and females')
data=df.groupby('country').sum()['suicides_no'].sort_values(ascending=False)
f,ax=plt.subplots(1,1,figsize=(10,20))
ax=sns.barplot(data.head(20),data.head(20).index)
plt.title('Country-wise suicide-count')
data=df.groupby('country').sum()['suicides_no'].sort_values(ascending=False)
f,ax=plt.subplots(1,1,figsize=(10,20))
ax=sns.barplot(data.tail(20),data.tail(20).index)
plt.figure(figsize=(10,8))
sns.scatterplot(x='gdp_per_capita',y='suicides_no',data=df)
plt.figure(figsize=(10,8))
df['gdp_for_year']=df['gdp_for_year'].str.replace(",","").astype("int64")
sns.scatterplot(x='gdp_for_year',y='suicides_no',data=df)
plt.figure(figsize=(10,8))
sns.scatterplot(x='year',y='suicides_no',data=df)