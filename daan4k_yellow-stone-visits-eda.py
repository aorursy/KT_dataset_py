import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from statistics import mean



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/ysnp.csv')

df.head(4)
dates = pd.to_datetime(df['Year/Month/Day'], infer_datetime_format=True)

df = df.drop('Year/Month/Day', axis=1)



df['day'] = dates.apply(lambda x: x.day)

df['month'] = dates.apply(lambda x: x.month)

df['year'] = dates.apply(lambda x: x.year)





df.head(4)
df.describe()
sns.lineplot(data=df[['Recreation Visits', 'MeanTemperature(F)']].groupby('MeanTemperature(F)').mean())
sns.lineplot(x="month", y="Recreation Visits",hue="year", data=df)
sns.lineplot(data=df[['Recreation Visits', 'TotalPrecipitation(In)']].groupby('TotalPrecipitation(In)').mean())
dfm = df.melt(var_name='columns')

g = sns.FacetGrid(dfm, col='columns',col_wrap=4,sharex=False,sharey=False)

g = g.map(sns.distplot, 'value')
sns.heatmap(df.corr())
sns.distplot(df["Recreation Visits"], bins=100)
df.dtypes