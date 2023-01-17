# First we must import the necessary modules for data manipulation and visual representation



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as matplot

import seaborn as sns

%matplotlib inline
# Now that we have all the modules loaded in we can now read the analytics csv

# file and store our dataset into a dataframe called "HR_DF"

HR_DF = pd.read_csv('../input/HR_comma_sep.csv')
# Even though the data is very clean in this dataset we should still

# check for any missing values.  To do this we enter:

HR_DF.isnull().any()
# Let's take a look at a quick overview of what exactly is in our dataset

HR_DF.head()
# That looks good but we should rename certain columns for better readability

HR_DF = HR_DF.rename(columns={'satisfaction_level': 'satisfaction',

                             'last_evaluation': 'evaluation',

                             'number_project': 'projectCount',

                             'average_montly_hours': 'AVG_MonthlyHours',

                             'time_spend_company': 'yearsAtCompany',

                             'Work_accident': 'workAccident',

                             'promotion_last_5years': 'promotion',

                             'sales': 'department',

                             'left': 'turnover'

                             })
# Now let's take a look at the table again

HR_DF.head()
# The table is now much more readible but the focus on this project is to see

# the employee turnover rate so let's move the variable "turnover"

# to the front of the table.

front = HR_DF['turnover']

HR_DF.drop(labels=['turnover'], axis=1,inplace = True)

HR_DF.insert(0, 'turnover', front)

HR_DF.head()
# Each row indicates an employee.  As we can see we have roughly 15,000

# employees and 10 feature that are observed.

HR_DF.shape
# Before we start any exploring the data we must know the "type" of

# our features.

HR_DF.dtypes
# Let's start with figuring out the turnover rate.  To do this we will use

# cross validation which will take the values in the 'turnover' column

# and divide it by the length of the dataset.

# As we can see about 76% of employees stayed and roughly 24% left the company.

turnover_rate = HR_DF.turnover.value_counts() / len(HR_DF)

turnover_rate
# The easiest way to show the statistical overview of the employees

# is to use the describe function

HR_DF.describe()
# But since we are focusing on turnover we can look at a summary

# of (Turnover vs. Non-turnover)

turnover_summary = HR_DF.groupby('turnover')

turnover_summary.mean()
# These first two histograms will show us what department the employees

# do in the company and where they are in salary.  We do the code

# as follows:

fig, axs = plt.subplots(ncols=2,figsize=(12,6))

x = sns.countplot(HR_DF['department'], ax=axs[0])

plt.setp(x.get_xticklabels(), rotation=45)

y = sns.countplot(HR_DF['salary'], ax=axs[1])

plt.tight_layout()

plt.show()
# Next lets look at employees who have had a work accident,

# promotion within the last 5 years, and has left the company.

# For these histograms:

# 0 = No

# 1 = Yes



fig, axs = plt.subplots(ncols=3,figsize=(12,6))

sns.countplot(HR_DF["workAccident"], ax=axs[0])

sns.countplot(HR_DF["promotion"], ax=axs[1])

sns.countplot(HR_DF["turnover"], ax=axs[2])

plt.tight_layout()

plt.show()
# Next we will take a look at employee satisfaction,

# employee evaluation, and average monthly hours:



fig, axs = plt.subplots(ncols=3,figsize=(12,6))

sns.distplot(HR_DF["satisfaction"], ax=axs[0])

sns.distplot(HR_DF["evaluation"], ax=axs[1])

sns.distplot(HR_DF["AVG_MonthlyHours"], ax=axs[2])

plt.tight_layout()

plt.show()
# Our final two charts show the number of projects given to

# employees and time spent with the company:



fig, axs = plt.subplots(ncols=2,figsize=(12,6))

axs[0].hist(HR_DF["projectCount"],bins=6)

axs[0].set_xlabel("Number of Projects")

axs[0].set_ylabel("Number of Employees")

axs[1].hist(HR_DF["yearsAtCompany"],bins=6,color='r')

axs[1].set_xlabel("Years at the Company")

axs[1].set_ylabel("Number of Employees")

plt.tight_layout()

plt.show()
# Let's take a look at a heat map to better correlate our data:

correlation = HR_DF.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')



plt.title('Correlation Between Features')



correlation