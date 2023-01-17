import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import seaborn as sns

import scipy.stats as stats

import statsmodels.api as sm

from statsmodels.stats.diagnostic import het_breuschpagan



import warnings

warnings.simplefilter('ignore', FutureWarning)
from sklearn.datasets import load_boston

boston = load_boston()

boston.keys()

data = pd.DataFrame(boston.data, columns=boston.feature_names)

data['target'] = boston.target

print(boston.DESCR)
def plot_residuals(results):

    '''

    Makes graphs that help us test assumptions 1, 2, and 3

    '''

    fig, ax = plt.subplots(ncols=2, figsize=(16, 6))

    sns.scatterplot(x=results.fittedvalues, y=results.resid, ax=ax[0]).set(xlabel='Predicted Values', ylabel='Residuals');

    ax[0].axhline(0, color='r')

    sm.qqplot(results.resid, fit=True, line='45', ax=ax[1]);

    ax[1].set_title('Skew: {}'.format(round(results.resid.skew(),3)));
fig, ax = plt.subplots(ncols=2, figsize=(12, 4))

sns.distplot(data['target'], ax=ax[0]).set_title('Target Skew. {}'.format(round(data['target'].skew(), 3)));

sns.distplot(np.log1p(data['target']), ax=ax[1]).set_title('Log of Target. Skew: {}'.format(round(np.log1p(data['target']).skew(), 3)));
corr_order = abs(data.corr()['target']).sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(16, 10))

sns.heatmap(data[corr_order].corr(), annot=True);
sns.lmplot(x='LSTAT', y='target', data=data, order=1);
X = sm.add_constant(data['LSTAT'])

y = data['target']

results = sm.OLS(y, X).fit()

plot_residuals(results)
sns.lmplot(x='LSTAT', y='target', data=data, order=2);
data['LSTAT_log'] = np.log1p(data['LSTAT'])

X = sm.add_constant(data['LSTAT_log'])

results = sm.OLS(y, X).fit()

plot_residuals(results)
data['target_log'] = np.log1p(data['target'])

X = sm.add_constant(data['LSTAT_log'])

results = sm.OLS(data['target_log'], X).fit()

plot_residuals(results)
data = data.drop(columns=['LSTAT', 'target'])
sns.lmplot(x='RM', y='target_log', data=data, order=1);
X = sm.add_constant(data['RM'])

results = sm.OLS(data['target_log'], X).fit()

plot_residuals(results)
data['RM_log'] = np.log1p(data['RM'])

X = sm.add_constant(data['RM_log'])

results = sm.OLS(y, X).fit()

plot_residuals(results)
data = data.drop(columns=['RM_log'])
X = sm.add_constant(data[['LSTAT_log','RM']])

results = sm.OLS(data['target_log'], X).fit()

plot_residuals(results)
results.summary()
print((1.01**-0.564 - 1)*100)

print((1.01**-0.472 - 1)*100)
print((np.exp(0.053)-1)*100)

print((np.exp(0.124)-1)*100)