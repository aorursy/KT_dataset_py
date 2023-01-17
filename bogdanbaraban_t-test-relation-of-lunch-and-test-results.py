import pandas as pd

from math import sqrt

from numpy import mean

from scipy.stats import sem

from scipy.stats import t
data=pd.read_csv('../input/StudentsPerformance.csv')

data['overall_score'] = data[['math score', 'reading score', 'writing score']].mean(axis=1)

print(data.describe())

data.head()
print(data.groupby(['lunch'])['overall_score'].mean())

print('====='*10)

print(data.groupby(['lunch'])['overall_score'].count())
from scipy.stats import ttest_ind

free_lunch_mean = data[(data["lunch"] == 'free/reduced')]

standard_lunch_mean = data[(data["lunch"] == 'standard')]



print('t=%.3f, p=%.3f' %(ttest_ind(standard_lunch_mean['overall_score'], 

                                   free_lunch_mean['overall_score'])))



print('With free Lunch:', data[(data["lunch"] == 'free/reduced')].overall_score.mean())

print('With standard Lunch:', data[(data["lunch"] == 'standard')].overall_score.mean())
# function for calculating the t-test for two samples

def ttest(data1, data2, alpha):

    # calculate means

    mean1, mean2 = mean(data1), mean(data2)

    # calculate standard errors

    se1, se2 = sem(data1), sem(data2)

    # standard error on the difference between the samples

    sed = sqrt(se1**2 + se2**2)

    # calculate the t statistic

    t_stat = (mean1 - mean2) / sed

    # degrees of freedom

    df = len(data1) + len(data2) - 2

    # calculate the critical value

    cv = t.ppf(1.0 - alpha, df) # at what point is 1-alpha percentile

    # calculate the p-value

    p = (1.0 - t.cdf(abs(t_stat), df)) * 2 # cdf - Cumulative Distribution Function

    # return everything

    return t_stat, df, cv, p
# generate two independent samples

data1 = data[data['lunch']=='standard']['overall_score']

data2 = data[data['lunch']=='free/reduced']['overall_score']

# calculate the t test

alpha = 0.05

t_stat, df, cv, p = ttest(data1, data2, alpha)

print('t=%.3f, degrees of freedom=%d, cv=%.3f, p=%.3f' %(t_stat, df, cv, p))



# interpret via critical value

if abs(t_stat) <= cv:

    print('Accept null hypothesis that the means are equal.')

else:

    print('Reject the null hypothesis that the means are equal.')

# interpret via p-value

if p > alpha:

    print('Accept null hypothesis that the means are equal.')

else:

    print('Reject the null hypothesis that the means are equal.')