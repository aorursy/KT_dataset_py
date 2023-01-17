# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats
df = pd.read_csv("../input/bank-additional-full.csv", sep=';')

df.head()
df.groupby('marital')['age'].mean()['single']
df[(df['marital'] == 'single') | (df['marital'] == 'divorced')]['age'].mean()
df[df['y'] == 'no']['day_of_week'].value_counts().keys()[0]
sns.heatmap(pd.crosstab(df['marital'], df['y']), cmap="BuPu", annot=True, cbar=True);
chi2_contingency(pd.crosstab(df['marital'], df['y']))
default_y_crosstab = pd.crosstab(df['default'], df['y'])



chi2_contingency(default_y_crosstab)
sns.heatmap(default_y_crosstab, cmap="BuPu", annot=True, cbar=True);
plt.figure(figsize=(10, 10))

df.groupby('education')['age'].mean().plot(kind='bar') 

plt.ylabel('Age')

plt.show();
stats.spearmanr(df.groupby('education')['age'].mean()[df['education']], df['education'])
stats.pearsonr(df['duration'], df['age'])
df['housing_bin']=(df['housing']=='yes').astype(int)

df.head()
stats.spearmanr(df['housing_bin'], df['education'])
df.head()
df['y_bin']=(df['y']=='yes').astype(int)
df.corr(method='pearson')
sns.heatmap(df.corr(method='pearson'));
print("Job:", stats.spearmanr(df['y_bin'], df['job']))

print("Marital:", stats.spearmanr(df['y_bin'], df['marital']))

print("Education:", stats.spearmanr(df['y_bin'], df['education']))

print("Loan:", stats.spearmanr(df['y_bin'], df['loan']))

print("Contact:", stats.spearmanr(df['y_bin'], df['contact']))

print("Month:", stats.spearmanr(df['y_bin'], df['month']))

print("Day_of_week:", stats.spearmanr(df['y_bin'], df['day_of_week']))

print("Poutcome:", stats.spearmanr(df['y_bin'], df['poutcome']))