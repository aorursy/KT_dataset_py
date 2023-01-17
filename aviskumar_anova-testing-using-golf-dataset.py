import pandas as pd

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

%matplotlib inline

from scipy.stats import ttest_1samp,wilcoxon,ttest_ind,shapiro,levene,mannwhitneyu,ttest_rel

import seaborn as sns

from statsmodels.stats.power import ttest_power
inp_data=pd.read_csv('../input/dataset-for-anova-testing/SM4-Golf.csv')
inp_data.head()
inp_data.shape
inp_data.info()
inp_data.describe()
sns.distplot(inp_data['Current'])
sns.distplot(inp_data['New'])
group1=inp_data['Current']

group2=inp_data['New']
t_statistic,p_value=ttest_ind(group1,group2)

print(t_statistic,p_value)
t_statistic,p_value=ttest_1samp(group2-group1,0)
print(t_statistic,p_value)
t_statistic,p_value=mannwhitneyu(group1,group2)
print(t_statistic,p_value)
t_statistic,p_value = ttest_rel(group2,group1)

print(t_statistic,p_value)
z_statistic,p_value=wilcoxon(group2-group1)

print(t_statistic,p_value)
levene(group1,group2)
shapiro(group2)
group1_var=np.var(group1)

group2_var=np.var(group2)

pooled_SD=np.sqrt(((39*group1_var)+(39*group2_var))/(40+40-2))

print(pooled_SD)
power_of_test=ttest_power(pooled_SD, nobs=40, alpha=0.05, alternative="two-sided")

power_of_test
n=40

sample_mean_current=np.mean(group1)
sigma1=np.std(group1)/np.sqrt(n)
conf_interval_Current=stats.t.interval(0.95,   #confidence level is 95%

                              df=n-1,      #degree of freedom

                              loc=sample_mean_current,   #sample mean of group1

                              scale=sigma1)    #SD estimate

print(conf_interval_Current)
n=40

sample_mean_new=np.mean(group2)
sigma2=np.std(group2)/np.sqrt(n)
conf_interval_new=stats.t.interval(0.95,   #confidence level is 95%

                              df=n-1,      #degree of freedom

                              loc=sample_mean_new,   #sample mean of group2

                              scale=sigma2)    #SD estimate

print(conf_interval_new)