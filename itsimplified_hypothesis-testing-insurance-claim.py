import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import kurtosis, skew, stats

from math import sqrt

from numpy import mean, var
data = pd.read_csv("../input/insurance2.csv")

print(data.head())
print("Summary Statistics of Medical Costs")

print(data['charges'].describe())

print("skew:  {}".format(skew(data['charges'])))

print("kurtosis:  {}".format(kurtosis(data['charges'])))

print("missing charges values: {}".format(data['charges'].isnull().sum()))

print("missing smoker values: {}".format(data['smoker'].isnull().sum()))
f, axes = plt.subplots(1, 2)

sns.kdeplot(data['charges'], bw=10000, ax=axes[0])

sns.boxplot(data['charges'], ax=axes[1])

plt.show()
#prepare our 2 groups to test

smoker = data[data['smoker']==1]

non_smoker = data[data['smoker']==0]
plt.title('Distribution of Medical Costs for Smokers Vs Non-Smokers')

ax = sns.kdeplot(smoker['charges'], bw=10000, label='smoker')

ax = sns.kdeplot(non_smoker['charges'], bw=10000, label='non-smoker')

plt.show()
plt.title('Distribution of Medical Costs for Smokers Vs Non-Smokers')

ax = sns.boxplot(x="smoker", y="charges", data=data)
statistic, pvalue = stats.ttest_ind(non_smoker['charges'], smoker['charges'], equal_var = False)

print("2 sample, 2 sided t-test pvalue:  {} t-stat: {}".format(pvalue,statistic))
# function to calculate Cohen's d for independent samples

def cohend(d1, d2):

	n1, n2 = len(d1), len(d2)

	s1, s2 = var(d1, ddof=1), var(d2, ddof=1)

	s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

	u1, u2 = mean(d1), mean(d2)

	return (u1 - u2) / s

	

d = cohend(smoker['charges'], non_smoker['charges'])

print("cohen's d:  {}".format(d))
plt.title("BMI Versus Charges")

ax = sns.scatterplot(x="bmi", y="charges", data=data)

plt.show()
data.bmi.corr(data.charges)
def corr_converge(data=data):

    for i in range(0,60000,1000):

        data_new = data[data['charges'] >= i]

        print("lower bound: {} \t correlation coefficient: {} \t number of observations: {}".format(i,data_new.bmi.corr(data_new.charges),len(data_new)))

        pass

    

corr_converge()
data_new = data[data['charges']>=21000]

plt.title("BMI Versus Charges Greater Than 21000")

ax = sns.scatterplot(x="bmi", y="charges", data=data_new)

plt.show()
data_new.bmi.corr(data_new.charges)