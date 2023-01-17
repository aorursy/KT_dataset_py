%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data1,data2 = "../input/presidents_one.csv", "../input/presidents_two.csv"
df1 = pd.read_csv(data1)

df2 = pd.read_csv(data2)
df2['Petitions Pending'] = df2['Petitions Pending (Commutations)'] + df2['Petitions Pending (Pardons)']

df2.drop(['Petitions Pending (Commutations)', 'Petitions Pending (Pardons)'], axis=1, inplace=True)
df2['Petitions Received'] = df2['Petitions Received (Pardons)'] + df2['Petitions Received (Commutations)']

df2.drop(['Petitions Received (Pardons)', 'Petitions Received (Commutations)'], axis=1, inplace=True)
df2['Petitions Denied'] = df2['Petitions Denied (Pardons)'] + df2['Petitions Denied (Commutations)']

df2.drop(['Petitions Denied (Commutations)', 'Petitions Denied (Pardons)'], axis=1, inplace=True)
df2['Petitions Closed Without Presidential Action'] = df2['Petitions Closed Without Presidential Action (Pardons)'] + df2['Petitions Closed Without Presidential Action (Commutations)']

df2.drop(['Petitions Closed Without Presidential Action (Pardons)', 'Petitions Closed Without Presidential Action (Commutations)'], axis=1, inplace=True)
df2['Petitions Denied or Closed Without Presidential Action'] = df2['Petitions Denied or Closed Without Presidential Action (Pardons)'] + df2['Petitions Denied or Closed Without Presidential Action (Commutations)']

df2.drop(['Petitions Denied or Closed Without Presidential Action (Pardons)', 'Petitions Denied or Closed Without Presidential Action (Commutations)'], axis=1, inplace=True)
df1.drop(['Respites'], axis=1, inplace=True)
# Now the columns match I can combine them together

df = pd.concat([df1,df2])
df.head()
## I want to plot the number of Petitions Received through time, color coded by the president

col = sns.color_palette("Set1", 20)

g = sns.lmplot(data=df, x='Fiscal Year', y="Petitions Received", hue='President', fit_reg=False, palette=col, size=5)

g.set(xlim=(1890,2020), title="Petitions Received through History")
## I want to plot the number of Petitions Received through time, color coded by the president

col = sns.color_palette("Set1", 20)

df['Percent Granted'] = df['Petitions Granted']/df['Petitions Received']

g = sns.lmplot(data=df, x='Fiscal Year', y='Percent Granted', hue='President', fit_reg=False, palette=col, size=5)

g.set(xlim=(1890,2020), title="Percent of Received Petitions Granted")
df_group = df.groupby('President').mean()
g = sns.barplot(data=df_group, y='Percent Granted', x=df_group.index, order=list(df['President'].unique()))

g.set_xticklabels(labels=list(df['President'].unique()), rotation=90)

g.set_title('Percent of Received Petitions Granted')
df_group = df.groupby('President').sum()
df_group.head()
g = sns.barplot(data=df_group, y='Petitions Granted', x=df_group.index, order=list(df['President'].unique()))

g.set_xticklabels(labels=list(df['President'].unique()), rotation=90)

g.set_title('Total number of Granted Petitions')