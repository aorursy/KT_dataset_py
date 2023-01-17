import numpy as np 

import pandas as pd 

import seaborn as sns

import pylab as plt

import scipy.stats



from IPython.display import display



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Read the .csv file

house_train = pd.read_csv("../input/train.csv")

house_train.head().T[:30]
house_train['HeatingQC'].dtype
set(house_train['HeatingQC'].values)
house_train['HeatingQC'].isnull().sum()
house_train['BldgType'].dtype
set(house_train['BldgType'].values)
house_train['BldgType'].isnull().sum()
# Create a random sample

sample = house_train.sample(300)
# By default the null hypothesis for one variable is 

# "the categories are assumed to be equally likely".
# A chi-square test for BldgType

scipy.stats.chisquare(pd.factorize(sample['BldgType'])[0])
# pvalue < 0.05, reject the null hypothesis
# A chi-square test for RoofStyle

scipy.stats. chisquare(pd.factorize(sample['HeatingQC'])[0])
# pvalue > 0.05, accept the null hypothesis
# The null hypothesis for two variables is 

# "the variable HeatingQC and the variable BldgType are independent".
# Build the crosstable sums (contingency table) of each category-relationship

cross_table = pd.crosstab(sample['HeatingQC'], sample['BldgType'])

cross_table
# A chi-square test for independence of variables in a contingency table

chi2, p, dof, ex = scipy.stats.chi2_contingency(cross_table)

chi2, p, dof
# We can't belive in the test results, because there are lots of small values (<5)

# in the contingency table
plt.figure(figsize=(15,10))

sns.countplot(x="BldgType", data=sample,

              facecolor=(0, 0, 0, 0), linewidth=7,

              edgecolor=sns.color_palette("Set1", 7))

plt.title('Sample Distribution of "BldgType" Categories', fontsize=20);
plt.figure(figsize=(15,10))

sns.countplot(x="HeatingQC", data=sample,

              facecolor=(0, 0, 0, 0), linewidth=7,

              edgecolor=sns.color_palette("Set1", 7))

plt.title('Sample Distribution of "HeatingQC" Categories', fontsize=20);
plt.figure(figsize=(15,10))

sns.countplot(y="BldgType", hue="HeatingQC", data=sample, palette='Set1')

plt.legend(loc=4)

plt.title('Sample Distribution of "BldgType" Categories Grouped by "HeatingQC"', 

          fontsize=20);