# importing modules

import pandas as pd



# Read datasets/yearly_deaths_by_clinic.csv into yearly

yearly = pd.read_csv('../input/datacamp-projects/Dr. Semmelweis and the Discovery of Handwashing/datasets/yearly_deaths_by_clinic.csv')



# Print out yearly

print(yearly)
# Calculate proportion of deaths per no. births

yearly['proportion_deaths'] = yearly.deaths.divide(yearly.births)



# Extract clinic 1 data into yearly1 and clinic 2 data into yearly2

yearly1 = yearly[yearly.clinic == 'clinic 1']

yearly2 = yearly[yearly.clinic == 'clinic 2']



# Print out yearly1

print(yearly1)
# This makes plots appear in the notebook

%matplotlib inline



# Plot yearly proportion of deaths at the two clinics

ax = yearly1.plot(x='year', y='proportion_deaths', label='clinic 1')

yearly2.plot(x='year', y='proportion_deaths', label='clinic 2', ax=ax)

ax.set_ylabel('Proportion deaths')
# Read datasets/monthly_deaths.csv into monthly

monthly = pd.read_csv('../input/datacamp-projects/Dr. Semmelweis and the Discovery of Handwashing/datasets/monthly_deaths.csv', parse_dates=['date'])



# Calculate proportion of deaths per no. births

monthly['proportion_deaths'] = monthly.deaths.divide(monthly.births)



# Print out the first rows in monthly

print(monthly.head())
# Plot monthly proportion of deaths

ax = monthly.plot(x='date', y='proportion_deaths')

ax.set_ylabel('Proportion deaths')
# Date when handwashing was made mandatory

import pandas as pd

handwashing_start = pd.to_datetime('1847-06-01')



# Split monthly into before and after handwashing_start

before_washing = monthly[monthly.date < handwashing_start]

after_washing = monthly[monthly.date >= handwashing_start]



# Plot monthly proportion of deaths before and after handwashing

ax = before_washing.plot(x='date', y='proportion_deaths', label='Before handwashing')

after_washing.plot(x='date', y='proportion_deaths', label='After handwashing', ax=ax)

ax.set_ylabel('Proportion deaths')
# Difference in mean monthly proportion of deaths due to handwashing

before_proportion = before_washing.proportion_deaths

after_proportion = after_washing.proportion_deaths

mean_diff = after_proportion.mean() - before_proportion.mean() 

print(mean_diff)
from numpy import percentile

# A bootstrap analysis of the reduction of deaths due to handwashing

boot_mean_diff = []

for i in range(3000):

    boot_before = before_proportion.sample(frac=1, replace=True)

    boot_after = after_proportion.sample(frac=1, replace=True)

    boot_mean_diff.append(boot_after.mean() - boot_before.mean())



# Calculating a 95% confidence interval from boot_mean_diff 

confidence_interval = percentile(boot_mean_diff, [2.5, 97.5])

print(confidence_interval)
# The data Semmelweis collected points to that:

doctors_should_wash_their_hands = True