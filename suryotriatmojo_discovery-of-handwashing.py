# import modules

import pandas as pd

import matplotlib.pyplot as plt



# read the datasets

yearly = pd.read_csv('../input/yearly_deaths_by_clinic.csv')



# print out the datasets

print(yearly)
# calculate the proportion of deaths per number of births

yearly['proportion_deaths'] = yearly['deaths']/ yearly['births']



# Extract clinic 1 data into yearly1 and clinic 2 data into yearly2

yearly1 = yearly[yearly['clinic'] == 'clinic 1']

yearly2 = yearly[yearly['clinic'] == 'clinic 2']



# Print out yearly1 and 

print(yearly1)

print(yearly2)
# This makes plots appear in the notebook

%matplotlib inline



# Plot yearly proportion of deaths at the two clinics

plt.plot(yearly1['year'], yearly1['proportion_deaths'])

plt.plot(yearly2['year'], yearly2['proportion_deaths'])

plt.title('Proportion of Deaths in Clinic 1 & Clinic 2')

plt.xlabel('Year')

plt.ylabel('Proportion Deaths per Number of Births')

plt.legend(['clinic 1', 'clinic 2'])

plt.show();
# Read datasets/monthly_deaths.csv into monthly

monthly = pd.read_csv('../input/monthly_deaths.csv', parse_dates = ['date'])



# Calculate proportion of deaths per no. births

monthly['proportion_deaths'] = monthly['deaths']/ monthly['births']



# Print out the first 5 rows in monthly

print(monthly.head())
# Plot monthly proportion of deaths

plt.plot(monthly['date'], monthly['proportion_deaths'])

plt.title('Proportion of Deaths by Date')

plt.xlabel('Year')

plt.ylabel('Proportion Deaths per Number of Births')

plt.show();
# Date when handwashing was made mandatory

handwashing_start = pd.to_datetime('1847-06-01')



# Split monthly into before and after handwashing_start

before_washing = monthly[monthly['date'] < handwashing_start]

after_washing = monthly[monthly['date'] >= handwashing_start]



# Plot monthly proportion of deaths before and after handwashing

plt.plot(before_washing['date'], before_washing['proportion_deaths'])

plt.plot(after_washing['date'], after_washing['proportion_deaths'])

plt.title('After Handwashing Highlighted')

plt.xlabel('Year')

plt.ylabel('Proportion Deaths per Number of Births')

plt.legend(['Before Handwahing', 'After Handwashing'])

plt.show();
# Difference in mean monthly proportion of deaths due to handwashing

before_proportion = before_washing['proportion_deaths']

after_proportion = after_washing['proportion_deaths']

mean_diff = after_proportion.mean() - before_proportion.mean()



print('Proportion of Deaths Average Before Handwashing = {}'.format(before_proportion.mean()))

print('Proportion of Deaths Average After Handwashing = {}'.format(after_proportion.mean()))

print('Mean Different Before & After Handwashing = {}'.format(mean_diff))
# A bootstrap analysis of the reduction of deaths due to handwashing

boot_mean_diff = []

for i in range(3000):                                                    # data collection by drawn 3000 random data for bootstrap analysis

    boot_before = before_proportion.sample(frac= 1, replace = True)      # bootstrap sampled from before_proportion

    boot_after = after_proportion.sample(frac= 1, replace = True)        # bootstrap sampled from after_proportion

    boot_mean_diff.append(boot_after.mean() - boot_before.mean())

    

# Calculating a 95% confidence interval from boot_mean_diff 

confidence_interval = pd.Series(boot_mean_diff).quantile([0.025, 0.975])

confidence_interval
# The data Semmelweis collected points to that:

doctors_should_wash_their_hands = True