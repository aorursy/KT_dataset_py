import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import ttest_1samp,ttest_ind,levene,shapiro,iqr,mannwhitneyu,wilcoxon,iqr

from statsmodels.stats.power import ttest_power

import scipy.stats as stats
golf = pd.read_csv('../input/Golf.csv')
golf.head()
golf.tail()
golf.info()
golf.describe()
golf.hist()
golf.boxplot()
Current = golf.iloc[:,0]
Current.head()
iqr(Current, rng = (25,75))
meanC = Current.mean()

meanC
varC = Current.var()

varC
New = golf.iloc[:,1]
New.head()
iqr(New, rng = (25,75))
meanN = New.mean()

meanN

varN= New.var()

varN
shapiro(Current)
shapiro(New)
t_statistic,p_value = ttest_ind(Current,New)
print(t_statistic,p_value)
levene(Current,New)
Pooledstd = np.sqrt(((40-1)*varC+ (40-1)*varN)/(40+40-2))

Pooledstd
delta = (meanC - meanN)/Pooledstd

delta
print(ttest_power(delta, nobs = 40, alpha = 0.05, alternative = 'larger'))
print(ttest_power(delta, nobs = 72, alpha = 0.05, alternative = 'larger'))
u,p_value = mannwhitneyu(Current,New,alternative = 'greater')
print(u,p_value)
critical = stats.t.isf(0.05,76)

critical
delta1 = critical - t_statistic

delta1
print(ttest_power(delta1, nobs = 40, alpha = 0.05, alternative = 'larger'))
print(ttest_power(delta1, nobs = 56, alpha = 0.05, alternative = 'larger'))