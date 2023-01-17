# importing modules

import pandas as pd

import numpy as np



# Read datasets/yearly_deaths_by_clinic.csv into yearly

yearly = pd.read_csv('../input/yearly_deaths_by_clinic.csv')



# Print out yearly

print(yearly)
# Calculate proportion of deaths per no. births

yearly['proportion_deaths'] =  yearly['deaths'] / yearly['births']



# Extract clinic 1 data into yearly1 and clinic 2 data into yearly2

yearly1 = yearly[yearly.clinic == 'clinic 1']

yearly2 = yearly[yearly.clinic == 'clinic 2']



# Print out yearly1

yearly1

# This makes plots appear in the notebook

%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('ggplot')



# Plot yearly proportion of deaths at the two clinics

ax = yearly1.plot(x='year', y='proportion_deaths', label='clinic 1')

# using 'ax=ax' so both of the frames being plotted in the same grid

yearly2.plot(x='year', y='proportion_deaths', label='clinic 2', ax=ax)



ax.set_ylabel('Proportion deaths')



# print plot

plt.show()
# Read datasets/monthly_deaths.csv into monthly

monthly = pd.read_csv('../input/monthly_deaths.csv', parse_dates=['date'])



# Calculate proportion of deaths per no. births

monthly['proportion_deaths'] = monthly.deaths / monthly.births



# Print out the first rows in monthly

monthly.head()
# Plot monthly proportion of deaths

ax = monthly.plot(x='date', y='proportion_deaths', color='green')

ax.set_ylabel('Proportion deaths')

plt.show()
# Date when handwashing was made mandatory

import pandas as pd

handwashing_start = pd.to_datetime('1847-06-01')



# Split monthly into before and after handwashing_start

before_washing = monthly[monthly.date < handwashing_start]

after_washing = monthly[monthly.date >= handwashing_start]



# Plot monthly proportion of deaths before and after handwashing

ax = before_washing.plot(x='date', y='proportion_deaths', label='before washing')

after_washing.plot(x='date', y='proportion_deaths', label='after washing', ax=ax)

ax.set_ylabel('Proportion deaths')

plt.show()
# Difference in mean monthly proportion of deaths due to handwashing

before_proportion = before_washing['proportion_deaths']

after_proportion = after_washing['proportion_deaths']



mean_diff = after_proportion.mean() - before_proportion.mean()

mean_diff
# A bootstrap analysis of the reduction of deaths due to handwashing

boot_mean_diff = []

for i in range(3000):

    boot_before = before_washing.proportion_deaths.sample(frac=1, replace=True)

    boot_after = after_washing.proportion_deaths.sample(frac=1,  replace=True)

    boot_mean_diff.append( boot_before.mean() - boot_after.mean() )



# Calculating a 95% confidence interval from boot_mean_diff 

confidence_interval = pd.Series(boot_mean_diff).quantile([0.025, 0.975])

confidence_interval
