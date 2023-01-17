%matplotlib inline



# Importing necessary libraries to power ETL, EDA, and statistical analysis.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns; # advanced visualization

from scipy import stats # statistical operations



# changing graph style following best practices from Storyteling with Data from Cole Knaflic [7]

sns.set(font_scale=2.5)

sns.set_style("whitegrid")
# Readig the data and setting index column

# Making a copy for future use

machine_dataset = pd.read_csv(r"../input/Data - Thesis 2019 - G1 Data Studio.csv", index_col=0)

machine_dataset_original = machine_dataset.copy()
# Checking how the dataset looks like

machine_dataset.head()
# Getting summary statistics

machine_dataset.describe()
# Plotting box plot distributions for outlier detection

fig1 = sns.boxplot(data=machine_dataset, fliersize=15, width=0.5, color="white")

fig1 = plt.gcf()

fig1.set_size_inches(16, 10)

sns.despine(left=False)

sns.set_context("talk")

fig1.text(0.07, 0.5, 'Porosity', ha='center', va='center', rotation='vertical')

fig1.suptitle("Box plot of porosity for Machines A and B", y=0.93, x=0.348, fontsize=28);
# Normality test before outlier exclusion

# Shapiro-Wilk Normality Test



# Adding additional libraries to power the test

from numpy.random import seed

from numpy.random import randn

from scipy.stats import shapiro



# Seed the random number generator

seed(1)



# Generate univariate observations

data = 5 * randn(100) + 50



# Normality test

stat, p = shapiro(machine_dataset["Machine B"])

print('Statistics=%.3f, p=%.3f' % (stat, p))



# Interpretations of p-value

alpha = 0.05

if p > alpha:

    print('Sample looks Gaussian (fail to reject H0)')

else:

    print('Sample does not look Gaussian (reject H0)')
# Calculating Z score, detecting and removing outliers for both Machines

# Outliers were eliminated by overwriting them with mean value for each machine distribution

outliers_machineA = machine_dataset[["Machine A"]][(np.abs(stats.zscore(machine_dataset[["Machine A"]])) > 3).all(axis=1)].index

machine_dataset.loc[outliers_machineA, "Machine A"] = machine_dataset["Machine A"].mean()



outliers_machineB = machine_dataset[["Machine B"]][(np.abs(stats.zscore(machine_dataset[["Machine B"]])) > 3).all(axis=1)].index

machine_dataset.loc[outliers_machineB, "Machine B"] = machine_dataset["Machine B"].mean()
# Plotting distribution of Machine working data after outlier exclusion

fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)

fig.subplots_adjust(wspace=0.02)

fig2 = sns.distplot(a=machine_dataset["Machine A"], ax=axes[0])

fig3 = sns.distplot(a=machine_dataset["Machine B"], color="green", ax=axes[1])

fig.set_size_inches(16, 10)

fig.suptitle("Distribution of porosity after outliers exclusion", y=0.93, x=0.368, fontsize=28);

fig.text(0.09, 0.5, 'Porosity Percentage Frequency', ha='center', va='center', rotation='vertical');
# Normality test after outlier exclusion



# Generate univariate observations

data = 5 * randn(100) + 50



# Normality test

stat, p = shapiro(machine_dataset["Machine B"])

print('Statistics=%.3f, p=%.3f' % (stat, p))



# Interpretation of p-value

alpha = 0.05

if p > alpha:

    print('Sample looks Gaussian (fail to reject H0)')

else:

    print('Sample does not look Gaussian (reject H0)')
# QQ Plot technique for graphical normality assesment 

from numpy.random import seed

from numpy.random import randn

from statsmodels.graphics.gofplots import qqplot



# Seed the random number generator

seed(1)



# QQ plot design

fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)

fig2 = qqplot(machine_dataset["Machine A"], line='s', ax=axes[0])

fig3 = qqplot(machine_dataset["Machine B"], line='s', ax=axes[1])

axes[0].set_title('Machine A')

axes[1].set_title('Machine B')

axes[0].set_ylabel('')

axes[1].set_ylabel('')

axes[0].set_xlabel('')

axes[1].set_xlabel('')

fig.subplots_adjust(wspace=0.05)

fig.set_size_inches(16, 10)

fig.suptitle("Quantile-Quantile plot for Machines A and B", y=0.97, x=0.36, fontsize=28);

fig.text(0.5, 0.04, 'Theoretical quantiles', ha='center', va='center');
# For optimal user of seaborn library we melt the original DataFrame for "tidy data" configuration

# Visualizing confidence interval overlap at 95% confidence level

machine_dataset_melt = pd.melt(machine_dataset_original, value_vars=["Machine A", "Machine B"])

fig = sns.pointplot(x="value", y="variable", data=machine_dataset_melt, join=False, capsize=0.1)

fig = plt.gcf()

fig.set_size_inches(16, 8)

plt.title("95% Confidence interval for sample means", y=1.02, x=0.3, fontsize=28);

plt.ylabel("");

plt.xlabel("Sample mean");

sns.despine(left=False)

sns.set_context("talk")
# T-test using statsmodel considering equal variance and equal n samples

stats.ttest_ind(machine_dataset_original["Machine A"], machine_dataset_original["Machine B"], equal_var = False)
# Generating the Reliability equation over the 10 years after manufacturing

year = list(range(1,12))



reliability_function = [0.999]

for i in year:

    try:

        reliability_function.append((0.999)*pow(0.99, year[i]-1))

    except:

        pass
# Converting lists into dictionaries for DataFrame

reliability_dict = {"Reliability":reliability_function, "Year":list(range(11))}

reliability = pd.DataFrame(reliability_dict)

fig, axes = plt.subplots(nrows=1, ncols=1)

fig1 = sns.scatterplot(x="Year", y="Reliability", data=reliability, palette="cmap", legend=False, ax=axes, s=200);



# Add text annotation

fig1.text(0.2, 1, "R(0) = 0.999", horizontalalignment='left', fontsize=14, color='black', weight='semibold');

fig1.text(10, 0.91, "R(10) = 0.903", horizontalalignment='center', fontsize=14, color='black', weight='semibold');

fig.set_size_inches(20, 5)

fig.suptitle("Reliability decay over time", x=0.213, y=0.95);
# Calculating the probability for each year using binomial cdf (using pmf)

from scipy.stats.distributions import binom

reliability["Probability"] = binom.pmf(5, 5, reliability["Reliability"])

reliability
# Resultant probability of parts to be in working condition by year 10

print("Reliability after 10 years = {0:0.2f}%".format(reliability.Probability.product()*100))