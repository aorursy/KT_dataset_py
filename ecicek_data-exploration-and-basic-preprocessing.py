import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/startup-investments-crunchbase/investments_VC.csv',encoding = 'unicode_escape')

df.head()
df.info()
df.tail()
df.dropna(how='all',inplace=True)
df[' market '].isna().sum()
len(df[' market '].unique())
df[' market '].value_counts().head(20)
sns.distplot(df[' market '].value_counts(),hist=False)
sum(df[' market '].value_counts()>500)
# 'funding_total_usd' column : this column is the sum of all fundings collected from one or more channels

# However, most of the values are string type and needs to be numeric (float)

df[' funding_total_usd '].apply(type).value_counts()
#let's create a new column named 'total_funding' and fill it with the sum all the fundings collected



funding_channels=['seed','venture', 'equity_crowdfunding', 'undisclosed', 'convertible_note','debt_financing', 'angel', 'grant', 'private_equity', 'post_ipo_equity', 'post_ipo_debt', 'secondary_market', 'product_crowdfunding']



df['total_funding']=0

for c in funding_channels:

    df['total_funding']=df['total_funding']+df[c]



df[[' funding_total_usd ','total_funding']].head()
len(df[df['total_funding']==0])/len(df)
sns.distplot(df[(df['total_funding']>0)]['total_funding'],hist=False)
df['total_funding'].describe()
df['status'].value_counts()
df['status'].isna().sum()/len(df)
len(df['country_code'].unique())
df['country_code'].isna().sum()
df['founded_year'].describe()
#percentage of NaN values in Founded Year Column

df['founded_year'].isna().sum()/len(df)
#distribution of founded years

sns.distplot(df['founded_year'],hist=False)
df2=df[['status','founded_year','total_funding']]

df2.head()
g = sns.pairplot(df2[df2['total_funding']<1000000],markers='.')
df.groupby(['status'])['total_funding'].describe()
sns.boxplot(x="status", y="total_funding", data=df)