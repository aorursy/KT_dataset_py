# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read datasets/yearly_deaths_by_clinic.csv into yearly

yearly = pd.read_csv("/kaggle/input/survey-data/yearly_deaths_by_clinic.csv")



# Print out yearly

yearly
# Calculate proportion of deaths per no. births

yearly["proportion_deaths"] = yearly["deaths"] / yearly["births"]



# Extract clinic 1 data into yearly1 and clinic 2 data into yearly2

yearly1 = yearly[yearly["clinic"] == "clinic 1"]

yearly2 = yearly[yearly["clinic"] == "clinic 2"]



# Print out yearly1

yearly1
# This makes plots appear in the notebook

%matplotlib inline



# Plot yearly proportion of deaths at the two clinics

ax = yearly1.plot(x="year", y="proportion_deaths", label="Clinic 1")

yearly2.plot(x="year", y="proportion_deaths", label="Clinic 2", ax=ax)

ax.set_ylabel("Proportion deaths")
# Read datasets/monthly_deaths.csv into monthly

monthly = pd.read_csv("/kaggle/input/survey-data/monthly_deaths.csv", parse_dates=["date"])



# Calculate proportion of deaths per no. births

monthly["proportion_deaths"] = monthly["deaths"] / monthly["births"]



# Print out the first rows in monthly

monthly.head()
# effect of handwashing

ax = monthly.plot(x="date", y="proportion_deaths")

ax.set_ylabel("Proportion deaths")
# Date when handwashing was made mandatory

import pandas as pd

handwashing_start = pd.to_datetime('1847-06-01')



# Split monthly into before and after handwashing_start

before_washing = monthly[monthly["date"] < handwashing_start]

after_washing = monthly[monthly["date"] >= handwashing_start]



# Plot monthly proportion of deaths before and after handwashing

ax = before_washing.plot(x="date", y="proportion_deaths",

                         label="Before handwashing")

after_washing.plot(x="date", y="proportion_deaths",

                   label="After handwashing", ax=ax)

ax.set_ylabel("Proportion deaths")
before_proportion = before_washing["proportion_deaths"]

after_proportion = after_washing["proportion_deaths"]

mean_diff = after_proportion.mean() - before_proportion.mean()

mean_diff
# A bootstrap analysis of the reduction of deaths due to handwashing

boot_mean_diff = []

for i in range(3000):

    boot_before = before_proportion.sample(frac=1, replace=True)

    boot_after = after_proportion.sample(frac=1, replace=True)

    boot_mean_diff.append( boot_after.mean() - boot_before.mean() )



# Calculating a 95% confidence interval from boot_mean_diff 

confidence_interval = pd.Series(boot_mean_diff).quantile([0.025, 0.975])

confidence_interval