# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
df = pd.read_csv("../input/Pisa mean perfromance scores 2013 - 2015 Data.csv")
df.head()
df = df.iloc[:1161,:].pivot(index="Country Name",columns="Series Name", values="2015 [YR2015]")
df.replace('', np.NaN)
df = df.replace(r'\s+',np.nan,regex=True).replace('..',np.nan)
df
df = df.dropna()
df.head()
df['mean'] = df.mean()
df = df.astype(float)
mean = []

for j in range(0, len(df)):

    k = 0

    for i in range(0,len(df.columns)-2):

        k+= float(df.iloc[j,i])

    mean += [k/8]

print(mean)
df['mean'] = df.iloc[:,:-2].mean(axis=1)
df.head()
df.sort_values(by='mean', ascending=False, na_position='first').head()
df['Ranking'] = df["mean"].rank(ascending=False)
best_countries = df.sort_values(by='Ranking', ascending=True, na_position='first')
best_15 = best_countries.head(15)
best_5 = best_countries.head(5)
best_5 = best_5.reset_index()
best_5
df_best_5 = pd.melt(best_5, id_vars=['Country Name'])
df_best_5.head()
sns.factorplot(x='Country Name', y='value',hue="Series Name", data=df_best_5, kind='bar', size=10)