import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

from datetime import datetime

import seaborn as sns

%matplotlib inline
df = []

df = pd.read_csv('../input/IT_motives_2013.csv')

df.head()
df.columns = df.columns.str.replace('\/','') 

df.head()
df = df.drop(df[df.StateUTs == 'Total (UTs)'].index)

df = df.drop(df[df.StateUTs == 'Total (States)'].index)

df = df.drop(df[df.StateUTs == 'Total (All India)'].index)
plt.figure(figsize=(70,30))

sns.set(font_scale=6)

sns.barplot(x='StateUTs', y='Total', data = df)

plt.title('Cyber Crimes - Total')

plt.ylabel('Total')

plt.xticks(rotation=90)
plt.figure(figsize=(70,30))

sns.set(font_scale=6)

yCol = 'Revenge Settling scores'

sns.barplot(x='StateUTs', y = yCol, data = df)

plt.title('Cyber Crimes - ' + yCol)

plt.ylabel('Total')

plt.xticks(rotation=90)
plt.figure(figsize=(70,30))

sns.set(font_scale=6)

yCol = 'Greed Money'

sns.barplot(x='StateUTs', y = yCol, data = df)

plt.title('Cyber Crimes - ' + yCol)

plt.ylabel('Total')

plt.xticks(rotation=90)
plt.figure(figsize=(70,30))

sns.set(font_scale=6)

yCol = 'Extortion'

sns.barplot(x='StateUTs', y = yCol, data = df)

plt.title('Cyber Crimes - ' + yCol)

plt.ylabel('Total')

plt.xticks(rotation=90)
plt.figure(figsize=(70,30))

sns.set(font_scale=6)

yCol = 'Cause Disrepute'

sns.barplot(x='StateUTs', y = yCol, data = df)

plt.title('Cyber Crimes - ' + yCol)

plt.ylabel('Total')

plt.xticks(rotation=90)
plt.figure(figsize=(70,30))

sns.set(font_scale=6)

yCol = 'Prank Satisfaction of Gaining Control '

sns.barplot(x='StateUTs', y = yCol, data = df)

plt.title('Cyber Crimes - ' + yCol)

plt.ylabel('Total')

plt.xticks(rotation=90)
plt.figure(figsize=(70,30))

sns.set(font_scale=6)

yCol = 'Fraud Illegal Gain'

sns.barplot(x='StateUTs', y = yCol, data = df)

plt.title('Cyber Crimes - ' + yCol)

plt.ylabel('Total')

plt.xticks(rotation=90)
plt.figure(figsize=(70,30))

sns.set(font_scale=6)

yCol = 'Eve teasing Harassment'

sns.barplot(x='StateUTs', y = yCol, data = df)

plt.title('Cyber Crimes - ' + yCol)

plt.ylabel('Total')

plt.xticks(rotation=90)
df_new = df

df_new = df_new.drop(['Crime Head','Total'], axis = 1)

df_new = pd.melt(df_new, id_vars=['StateUTs','Year'], var_name = 'CrimeType')

df_new.head()
df_totalCrimeType = pd.DataFrame({'TotalCrimes' : df_new.groupby(['CrimeType']).value.sum()}).reset_index()

df_totalCrimeType
plt.figure(figsize=(70,30))

sns.set(font_scale=6)

sns.barplot(x='CrimeType', y='TotalCrimes', data=df_totalCrimeType)

plt.title('Cyber Crimes - Overall')

plt.xlabel('Type of Crime')

plt.ylabel('Total Crimes')

plt.xticks(rotation=90)