import os

print(os.listdir("../input"))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(15, 10)});
df = pd.read_csv("../input/labanumbaone/adult.csv")
df.head(15)
df.info()
df.describe().T
sns.boxplot(df['hours-per-week'])
sns.boxplot(df['educational-num'])
sns.boxplot(df['age'])
df.groupby('income')['age'].describe()
sns.countplot(y='age', hue='income', data=df);
df['fme'] = (df['education'] == 'Bachelors') | (df['education'] == 'Prof-school') | (df['education'] ==  'Assoc-acdm') | (df['education'] ==  'Assoc-voc') | (df['education'] ==  'Masters') | (df['education'] == 'Doctorate')

# pd.crosstab(df['income'], df['full_mid_education'])

df[df['income']=='>50K']['fme'].value_counts(normalize=True)
print('Married:\n',df[(df['gender'] == 'Male') & (df['relationship'] == 'Husband')]['income'].value_counts(normalize=True))

print('Unmarried:\n',df[(df['gender'] == 'Male') & (df['relationship'] != 'Husband')]['income'].value_counts(normalize=True))
mhpw=df['hours-per-week'].max()

mhpw
df["hours-per-week"][df["hours-per-week"] == mhpw].count()
df["income"][df["hours-per-week"] == mhpw].value_counts(normalize=True)
sns.heatmap(df[['hours-per-week','capital-gain']].corr(method='spearman'));
from scipy.stats import spearmanr, kendalltau

spearmanr(df['capital-gain'], df['hours-per-week'])
kendalltau(df['capital-gain'], df['hours-per-week'])
from scipy.stats import pointbiserialr

df['income num']=df['income'].map({'<=50K':0,'>50K':1})

df['income>50k'] = df['income'] == '>50K'

pointbiserialr(df['income num'], df['hours-per-week'])
f = df.groupby('education')['hours-per-week'].mean()

ax = sns.barplot(f.index, f.values,palette="Paired").set_xticklabels(f.index, rotation = 45)
df['income>50k'] = df['income'] == '>50K'
f = df.groupby('native-country')['income>50k'].mean().sort_values()

ax = sns.barplot(f.index, f.values).set_xticklabels(f.index, rotation = 90)
pd.crosstab(df['gender'], df['income'])
pd.crosstab(df['race'], df['income'])
df['income num']=df['income'].map({'<=50K':0,'>50K':1})

sns.heatmap(df[['capital-gain','income num']].corr(method='spearman'));
spearmanr(df['capital-gain'],df['income num'])