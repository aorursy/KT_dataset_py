# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)}); # you can change this if needed
df = pd.read_csv("../input/adult.csv")
print("mean value:", df.groupby('income')['age'].mean())

print("standart deviation:", df.groupby('income')['age'].std())

plt.figure(figsize=(15, 20))

sns.countplot(y='age', hue='income', data=df);
df['full_mid_education'] = (df['education'] == 'Bachelors') | (df['education'] == 'Prof-school') | (df['education'] ==  'Assoc-acdm') | (df['education'] ==  'Assoc-voc') | (df['education'] ==  'Masters') | (df['education'] == 'Doctorate')

pd.crosstab(df['income'], df['full_mid_education'])
print(df[(df['gender'] == 'Male') & (df['relationship'] == 'Husband')]['income'].value_counts(normalize=True))

print(df[(df['gender'] == 'Male') & (df['relationship'] == 'Unmarried')]['income'].value_counts(normalize=True))
max_hours_per_week = df["hours-per-week"].max()

print("maximum number of hours per week:", max_hours_per_week)

print("number of people who work so much:", df["hours-per-week"][df["hours-per-week"] == max_hours_per_week].count())

print("percentage of people working that much relative to their earnings:")

print(df["income"][df["hours-per-week"] == max_hours_per_week].value_counts(normalize=True))
from scipy.stats import pointbiserialr

df['income>50k'] = df['income'] == '>50K'

pointbiserialr(df['income>50k'], df['hours-per-week'])
fig = df.groupby('educational-num')['hours-per-week'].mean()

ax = sns.barplot(fig.index, fig.values,palette="Paired")
fig = df.groupby('native-country')['income>50k'].mean().sort_values()

ax = sns.barplot(fig.index, fig.values)
print('Male\n', df[df['gender'] == 'Male']['income'].value_counts(normalize=True))

print('Female\n', df[df['gender'] == 'Female']['income'].value_counts(normalize=True))