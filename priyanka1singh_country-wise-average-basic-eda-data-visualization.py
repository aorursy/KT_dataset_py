import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
col=['Country','Income Classification','Severe Wasting','Wasting','Overweight','Stunting','Underweight','U5_Population']

df=pd.read_csv("/kaggle/input/malnutrition-across-the-globe/country-wise-average.csv",names=col,skiprows=1)

df
df.info()
df.describe()
df.describe(include='object')
df.isnull().sum()
df.Country.value_counts().sort_values(ascending=False)
sns.distplot(df.Stunting)
df.Stunting=df.Stunting.fillna(df.Stunting.mean())
sns.distplot(df['Severe Wasting'])
df['Severe Wasting'].value_counts()
sns.boxplot(df['Severe Wasting'])
q1=df['Severe Wasting'].quantile(0.25)

q3=df['Severe Wasting'].quantile(0.75)

iqr=q3-q1

df=df[~((df['Severe Wasting']<(q1-iqr*1.5))|(df['Severe Wasting']>(q3+iqr*1.5)))]
sns.boxplot(df['Severe Wasting'])
df['Severe Wasting']=df['Severe Wasting'].fillna(df['Severe Wasting'].mean())
df['Severe Wasting'].isnull().sum()
sns.distplot(df['Wasting'])
sns.boxplot(df['Wasting'])
q1=df['Wasting'].quantile(0.25)

q3=df['Wasting'].quantile(0.75)

iqr=q3-q1

df=df[~((df['Wasting']<(q1-iqr*1.5))|(df['Wasting']>(q3+iqr*1.5)))]
df['Wasting'].value_counts()
df['Wasting']=df['Wasting'].fillna(df['Wasting'].mean())
sns.distplot(df['Overweight'])
sns.boxplot(df['Overweight'])
q1=df['Overweight'].quantile(0.25)

q3=df['Overweight'].quantile(0.75)

iqr=q3-q1

df=df[~((df['Overweight']<(q1-iqr*1.5))|(df['Overweight']>(q3+iqr*1.5)))]
sns.boxplot(df['Overweight'])
df['Overweight']=df['Overweight'].fillna(df['Overweight'].mean())
sns.distplot(df['Underweight'])
sns.boxplot(df['Underweight'])
q1=df['Underweight'].quantile(0.25)

q3=df['Underweight'].quantile(0.75)

iqr=q3-q1

df=df[~((df['Underweight']<(q1-iqr*1.5))|(df['Underweight']>(q3+iqr*1.5)))]
df['Underweight']=df['Underweight'].fillna(df['Underweight'].mean())
sns.countplot(df['Income Classification'])
df.groupby(['Income Classification'])['Overweight'].sum().plot.bar()
income0_wise_country=df[df['Income Classification']==0.0]

income1_wise_country=df[df['Income Classification']==1.0]

income2_wise_country=df[df['Income Classification']==2.0]

income3_wise_country=df[df['Income Classification']==3.0]
df['Income Classification'].value_counts()
income0_wise_country['Country']
plt.figure(figsize=(16,9))

income0_wise_country.groupby(['Country'])['Overweight','Underweight'].mean().plot.bar()
plt.figure(figsize=(12,8))

income1_wise_country.groupby(['Country'])['Overweight','Underweight'].mean().plot.bar()
plt.figure(figsize=(12,8))

income2_wise_country.groupby(['Country'])['Overweight','Underweight'].mean().plot.bar()

plt.show()
plt.figure(figsize=(12,8))

income3_wise_country.groupby(['Country'])['Overweight','Underweight'].mean().plot.bar()

plt.show()
#here you can figure out that australia has almost 0percent underweight and around 14% overweight population
plt.figure(figsize=(16,9))

income0_wise_country.groupby(['Country'])['Wasting','Severe Wasting'].mean().plot.bar()
income0_wise_country.groupby(['Country'])['Underweight'].max().sort_values(ascending=False).head(1)
income1_wise_country.groupby(['Country'])['Underweight'].max().sort_values(ascending=False).head(1)
income2_wise_country.groupby(['Country'])['Underweight'].max().sort_values(ascending=False).head(1)
income3_wise_country.groupby(['Country'])['Underweight'].max().sort_values(ascending=False).head(1)
income0_wise_country.groupby(['Country'])['Overweight'].max().sort_values(ascending=False).head(1)
income1_wise_country.groupby(['Country'])['Overweight'].max().sort_values(ascending=False).head(1)
income2_wise_country.groupby(['Country'])['Overweight'].max().sort_values(ascending=False).head(1)
income3_wise_country.groupby(['Country'])['Overweight'].max().sort_values(ascending=False).head(1)
income0_wise_country.groupby(['Country'])['Stunting'].max().sort_values(ascending=False).head(1)
income1_wise_country.groupby(['Country'])['Stunting'].max().sort_values(ascending=False).head(1)
income2_wise_country.groupby(['Country'])['Stunting'].max().sort_values(ascending=False).head(1)
income3_wise_country.groupby(['Country'])['Stunting'].max().sort_values(ascending=False).head(1)
income0_wise_country.groupby(['Country'])['Wasting'].max().sort_values(ascending=False).head(1)
income1_wise_country.groupby(['Country'])['Wasting'].max().sort_values(ascending=False).head(1)
income2_wise_country.groupby(['Country'])['Wasting'].max().sort_values(ascending=False).head(1)
income3_wise_country.groupby(['Country'])['Wasting'].max().sort_values(ascending=False).head(1)
income0_wise_country.groupby(['U5_Population','Country'])['Stunting'].max().sort_values(ascending=False).head(1)
income0_wise_country.groupby(['Country','U5_Population'])['Stunting'].max().sort_values(ascending=False).plot.bar()
income1_wise_country.groupby(['U5_Population','Country'])['Stunting'].max().sort_values(ascending=False).head(1)
income1_wise_country.groupby(['U5_Population','Country'])['Stunting'].max().sort_values(ascending=False).plot.bar()
income2_wise_country.groupby(['U5_Population','Country'])['Stunting'].max().sort_values(ascending=False).head(1)
income2_wise_country.groupby(['U5_Population','Country'])['Stunting'].max().sort_values(ascending=False).plot.bar()
income3_wise_country.groupby(['U5_Population','Country'])['Stunting'].max().sort_values(ascending=False).head(1)
income3_wise_country.groupby(['U5_Population','Country'])['Stunting'].max().sort_values(ascending=False).plot.bar()
income0_wise_country.groupby(['U5_Population','Country'])['Wasting'].max().sort_values(ascending=False).head(1)
income0_wise_country.groupby(['U5_Population','Country'])['Wasting'].max().sort_values(ascending=False).plot.bar()
income1_wise_country.groupby(['U5_Population','Country'])['Wasting'].max().sort_values(ascending=False).head(1)
income1_wise_country.groupby(['U5_Population','Country'])['Wasting'].max().sort_values(ascending=False).plot.bar()
income2_wise_country.groupby(['U5_Population','Country'])['Wasting'].max().sort_values(ascending=False).head(1)
income2_wise_country.groupby(['U5_Population','Country'])['Wasting'].max().sort_values(ascending=False).plot.bar()
income3_wise_country.groupby(['U5_Population','Country'])['Wasting'].max().sort_values(ascending=False).head(1)
income3_wise_country.groupby(['U5_Population','Country'])['Wasting'].max().sort_values(ascending=False).plot.bar()