import numpy as np

import pandas as pd

import re

import matplotlib.pyplot as plt

import seaborn as sns
dataset = pd.read_csv('/kaggle/input/mental-health-in-tech-survey/survey.csv')

# print(dataset.head())

print(dataset.columns)
print(dataset.Gender.unique())

dataset['Gender'] = dataset['Gender'].replace(to_replace = '^[mM]$|male', value='Male', regex=True)

dataset['Gender'] = dataset['Gender'].replace(to_replace = '^[Ff]$|[Ff]e[Mm]ale', value='Female', regex=True)

dataset = dataset[(dataset['Gender'] == 'Male')|(dataset['Gender'] == 'Female')]

print(dataset.Gender.unique())
column = dataset['Age']

print(np.mean(column))

print(np.min(column))

print(np.max(column))

dataset = dataset[dataset['Age'].between(12, 100)]

new_column = dataset['Age']

print(np.mean(new_column))

print(np.min(new_column))

print(np.max(new_column))
gender = dataset.groupby('Gender')['Timestamp'].count().reset_index()

print(gender)

x = gender['Gender']

y = gender['Timestamp']

sns.barplot(x, y)

plt.ylabel('Count')

plt.show()



x = dataset['Gender']

y = dataset['Age']

sns.violinplot(x, y)

plt.show()
dataset.rename(columns={'care_options': 'care',

                       'wellness_program': 'wellness',

                       'seek_help': 'help'}, inplace=True)

available = pd.get_dummies(dataset[['benefits', 'care', 'wellness', 'help', 'anonymity']])

available = available.T.groupby([s.split('_')[1] for s in available.T.index.values]).sum().T

available['Uncertain'] = available["Don't know"] + available['Not sure']

available = available[['No', 'Yes', 'Uncertain']]

available.rename(columns={'No': 'avai_no',

                         'Yes': 'avai_yes',

                         'Uncertain': 'avai_uncertain'}, inplace=True)

print(available.head())

dataset = dataset.merge(available, left_index=True, right_index=True)
from scipy.stats import ttest_ind



x = dataset['treatment']

y = dataset['avai_yes']

sns.barplot(x, y)

plt.xlabel('Have Sought Treatment')

plt.ylabel('Availability of Resources')

plt.show()



group_1 = dataset.loc[dataset['treatment'] == 'Yes', 'avai_yes']

group_2 = dataset.loc[dataset['treatment'] == 'No', 'avai_yes']

tstat, pval = ttest_ind(group_1, group_2)

print(pval)
from scipy.stats import f_oneway

from statsmodels.stats.multicomp import pairwise_tukeyhsd



x = dataset['work_interfere']

y = dataset['avai_yes']

sns.barplot(x, y)

plt.xlabel('Interference with Work')

plt.ylabel('Availability of Resources')

plt.show()



o = dataset.loc[dataset['work_interfere'] == 'Often', 'avai_yes']

r = dataset.loc[dataset['work_interfere'] == 'Rarely', 'avai_yes']

n = dataset.loc[dataset['work_interfere'] == 'Never', 'avai_yes']

s = dataset.loc[dataset['work_interfere'] == 'Sometimes', 'avai_yes']

pval = f_oneway(o, r, n, s).pvalue

print(pval)



v = np.concatenate([n, r, s, o])

labels = ['n']*len(n) + ['r']*len(r) + ['s']*len(s) + ['o']*len(o)

tukey_results = pairwise_tukeyhsd(v, labels, 0.05)

print(tukey_results)