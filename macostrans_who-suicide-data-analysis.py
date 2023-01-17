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
import seaborn as sns
import numpy as np
df = pd.read_csv(r"../input/who_suicide_statistics.csv")
df.head()
#Converting Population to in Lacs
df.population = df.apply(lambda x:x['population']/100000,axis=1)
df.head()
df.isna().any()
#No of Null values in population
df.population.isnull().count()-df.population.count()
import missingno as msno
msno.matrix(df)
df['suicides_no'].fillna(value=0,inplace=True)
df['suicides_no'] = df['suicides_no'].astype(int)
df.head()
df.drop(['population'],axis=1,inplace = True)
df['country'].value_counts().sort_values().head(30).plot.bar(figsize  = (17,9), title = "Suicide Amount for 30 Countries(Ascending)")
len(df['country'].unique())
print(min(df.year))
print(max(df.year))
print(df.groupby('sex')['suicides_no'].sum())
df.groupby('sex')['suicides_no'].sum().plot.bar(figsize=(12,7),title = "Male vs Female Overall Suicide comparison")
#understanding Age bins
df.age.unique()
print(df.groupby('age')['suicides_no'].sum())
df.groupby('age')['suicides_no'].sum().sort_index().plot.bar(figsize=(15,7),title = "Agewise Suicide comparison")
bins = [1978,1988,1998,2008,2016]
df['binned'] = pd.cut(df['year'], bins)
df.head()
print(df.groupby('binned')['suicides_no'].sum())
df.groupby('binned')['suicides_no'].sum().sort_index().plot.bar(figsize=(15,8),title = "Suicide comparison for 10 Yr Intervals")
#sns.barplot(x=df['binned'], y = )
data = df.groupby('binned')['suicides_no'].sum()
data
sns.barplot(x=data.index,y=data)
print(df[df.sex=='female'].groupby('binned')['suicides_no'].sum())
print(df[df.sex=='male'].groupby('binned')['suicides_no'].sum())
df.groupby(['sex','age','binned'], as_index=False)['suicides_no'].sum().head()
