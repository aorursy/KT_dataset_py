from scipy.stats import ttest_1samp,ttest_ind,ttest_rel

import numpy as np

import pandas as pd

diab = pd.read_csv("diabetes.csv")

diab.head()
age_mean = np.mean(diab['Age'])

age_mean
tval, pval = ttest_1samp(diab['Age'], 30)
print("p-values",pval)
print(tset)
if pval < 0.05:    # alpha value is 0.05 or 5%

    print(" we are rejecting null hypothesis")

else:

    print("we are accepting null hypothesis")
age_mean = np.mean(diab['Age'])

print("Mean of Age: ",age_mean)

bmi_mean = np.mean(diab['BMI'])

print("Mean of BMI",bmi_mean)
ttest,pval = ttest_rel(diab['Age'], diab['BMI'])

print(pval)
if pval<0.05:

    print("reject null hypothesis")

else:

    print("accept null hypothesis")
from statsmodels.stats import weightstats as stests
ztest ,pval = stests.ztest(diab['Age'], x2=None, value=156)

print(pval)

print(ztest)
if pval<0.05:

    print("reject null hypothesis")

else:

    print("accept null hypothesis")
ztest ,pval1 = stests.ztest(diab['Age'], x2= diab['BMI'], value=0,alternative='two-sided')

print(float(pval1))

print(ztest)
if pval<0.05:

    print("reject null hypothesis")

else:

    print("accept null hypothesis")
from scipy.stats import normaltest

data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]

stat, p = normaltest(data)

print('stat=%.3f, p=%.3f' % (stat, p))

if p > 0.05:

    print('Probably Gaussian')

else:

    print('Probably not Gaussian')