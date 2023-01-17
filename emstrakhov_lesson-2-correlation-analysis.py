import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set();
df = pd.read_csv('../input/cardio_train.csv', sep=';', index_col=0)
df.head().T
df.info()
df['age_years'] = df['age'] / 365.25 # возраст в годах

numeric = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']

sns.pairplot(df[numeric]);
sns.boxplot(df['age_years']);
sns.boxplot(df['height']);
sns.boxplot(df['weight']);
sns.boxplot(df['ap_hi']);
sns.boxplot(df['ap_lo']);
def outliers_indices(feature):

    '''

    Будем считать выбросами все точки, выходящие за пределы трёх сигм.

    '''

    mid = df[feature].mean()

    sigma = df[feature].std()

    return df[(df[feature] < mid - 3*sigma) | (df[feature] > mid + 3*sigma)].index
wrong_height = outliers_indices('height')

wrong_weight = outliers_indices('weight')

wrong_hi = outliers_indices('ap_hi')

wrong_lo = outliers_indices('ap_lo')



out = set(wrong_height) | set(wrong_weight) | set(wrong_hi) | set(wrong_lo)



print(len(out))
df.drop(out, inplace=True)
df[numeric].corr(method='spearman')
sns.heatmap(df[numeric].corr(method='spearman'));
from scipy.stats import pearsonr, spearmanr, kendalltau

r = pearsonr(df['height'], df['weight'])

print('Pearson correlation:', r[0], 'p-value:', r[1])
r_height_pressure = pearsonr(df['height'], df['ap_hi'])

print('Height vs. Pressure:', r_height_pressure)

r_weight_pressure = pearsonr(df['weight'], df['ap_hi'])

print('Weight vs. Pressure:', r_weight_pressure)
pd.crosstab(df['smoke'], df['cardio'])
sns.heatmap(pd.crosstab(df['smoke'], df['cardio']), 

            cmap="YlGnBu", annot=True, cbar=False);
from scipy.stats import chi2_contingency, fisher_exact

chi2_contingency(pd.crosstab(df['smoke'], df['cardio']))
fisher_exact(pd.crosstab(df['smoke'], df['cardio']))
from scipy.stats import pointbiserialr

pointbiserialr(df['cardio'], df['weight'])