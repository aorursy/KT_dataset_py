from scipy import stats 

import numpy as np
GPU1 = np.array([11,9,10,11,10,12,9,11,12,9,11,12,9,10,9])

GPU2 = np.array([11,13,10,13,12,9,11,12,12,11,12,12,10,11,13])



#Assumption: Both the datasets (GPU1 & GPU 2) are random, independent, parametric & normally distributed
from scipy.stats import ttest_1samp

from statsmodels.stats.power import ttest_power



t_stats, p_values = ttest_1samp(GPU1,0)

print(t_stats, p_values)
# Lets calculate the power of test also

from statsmodels.stats.power import ttest_power

(np.mean(GPU1)- 0 )/np.std(GPU1)
print(ttest_power(9.1019,nobs=15,alpha = 0.05,alternative = "two-sided"))
from scipy.stats import ttest_ind

t_stats,p_values = ttest_ind(GPU1,GPU2)

print(t_stats,p_values)
(np.mean(GPU1)-np.mean(GPU2))/np.sqrt(((15-1)*np.var(GPU1)+ (15-1)*np.var(GPU2))/15+15-2)
print(ttest_power(0.2885,nobs=15,alpha=0.05,alternative="two-sided"))
GPU3 = np.array([9,10,9,11,10,13,12,9,12,12,13,12,13,10,11])



#Assumption: Both the datasets (GPU1 & GPU 3) are random, independent, parametric & normally distributed
from scipy.stats import ttest_ind

t_stats,p_values = ttest_ind(GPU1,GPU3)

print(t_stats,p_values)
(np.mean(GPU1)-np.mean(GPU3))/np.sqrt(((15-1)*np.var(GPU1) + (15-1) * np.var(GPU3))/15+15-2)

print(ttest_power(0.1826,nobs = 15,alpha=0.05,alternative="two-sided"))
import numpy as np



e1 = np.array([1.595440,1.419730,0.000000,0.000000])

e2 = np.array([1.433800,2.079700,0.892139,2.384740])

e3 = np.array([0.036930,0.938018,0.995956,1.006970])



#Assumption: All the 3 datasets (e1,e2 & e3) are random, independent, parametric & normally distributed
# check the equality of variances and normality of various distribution

stats.levene(e1,e2,e3)
#e1.shape

#e2.shape

#e3.shape

stats.f_oneway(e1,e2,e3)
import numpy as np



d1 = [5, 8, 3, 8]

d2 = [9, 6, 8, 5]

d3 = [8, 12, 7, 2]

d4 = [4, 16, 7, 3]

d5 = [3, 9, 6, 5]

d6 = [7, 2, 5, 7]



df = np.array([d1, d2, d3, d4, d5, d6])
df.dtype
#converting into categorical dtype

import pandas as pd

d1 = pd.Categorical(d1)

d2 = pd.Categorical(d2)

d3 = pd.Categorical(d3)

d4 = pd.Categorical(d4)

d5 = pd.Categorical(d5)

d6 = pd.Categorical(d6)

lst = [d1,d2,d3,d4,d5,d6]

df = pd.DataFrame(lst) 

df
#stats.chi2_contingency(dice)



chi2, p, dof, ex = stats.chi2_contingency(df, correction=False)

print("chi2 :",chi2)

print("P_value: ",p)

print("degree of freedom:",dof)

print('\n')

print("ex :",ex)
import scipy.stats as sc

z_scores = sc.zscore(df)

z_scores
import scipy as sc

p_values = 1 - sc.special.ndtr(z_scores)

#p_value = sc.norm.pdf(abs(z_scores))

p_values
#z_scores.mean()

p_values.mean()
before= stats.norm.rvs(scale=30, loc=100, size=500) ## Creates a normal distribution with a mean value of 100 and std of 30

after = before + stats.norm.rvs(scale=5, loc=-1.25, size=500)
#before.shape

#after.shape

stats.ttest_rel(before,after)