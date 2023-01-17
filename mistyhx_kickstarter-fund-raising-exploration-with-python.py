import pandas as pd

from pandas import Series, DataFrame

import seaborn as sns 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns 

%matplotlib inline 

sns.set_style('whitegrid')
df = pd.read_csv('../input/ks-projects-201801.csv')
df.info()
df.describe()
df.head()
df['main_category'].describe()
df.isnull().sum()
state_color = ["#FD2E2E", "#E6E6E6", "#17B978", "#CFCBF1", "#4D4545", "#588D9C"]

sns.factorplot('state',data=df,kind='count', size=10,palette=state_color)
group_state = df.groupby(['state'])
df['state'].value_counts()
Success_Rate = 133956/378661

Success_Rate
sns.factorplot('main_category',data=df,kind='count',hue='state' ,size=15, palette=state_color)

group_maincategories = df.groupby(['main_category'])

group_maincategories['main_category'].count().reset_index(name='count').sort_values(['count'], ascending=False)
goal_fund = df['goal'].groupby(df['main_category'])

goal_fund.mean().reset_index(name='mean').sort_values(['mean'], ascending=False)
df['usd pledged'].describe()

#lets take out the null value so the average for usd pledged would be more precise 
df_tech = df[df['main_category']=='Technology']

df_tech.head()
df_tech['goal'].describe()
df_tech['usd pledged'].quantile(0.90)
df_tech[df_tech['usd pledged']>119712].count()