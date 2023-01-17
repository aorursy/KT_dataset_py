import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

print('All libraries have been imported.')
salaries_major_filepath = '../input/college-salaries/degrees-that-pay-back.csv'

salaries_major = pd.read_csv(salaries_major_filepath)



salaries_major.head()
# Rename cols.

salaries_major.columns = ['major', 'start', 'mid_career', 'delta_start_mid', 'mid_10p',

            'mid_25p', 'mid_75p', 'mid_90p']



salaries_major.head()
# Check dtypes for salary figures.

type(salaries_major.start[0])
salary_cols = ['start', 'mid_career', 'mid_10p', 'mid_25p', 'mid_75p', 'mid_90p']



for col in salary_cols:

    salaries_major[col] = salaries_major[col].str.replace('$', '') # remove dollar signs.

    salaries_major[col] = salaries_major[col].str.replace(',', '').astype(float) # remove commas and convert to floats.

    

salaries_major.head()
# Verify salary figures have been converted to floats.

type(salaries_major.start[0])
salaries_major.describe()
# Top 5 starting salaries.

salaries_major.sort_values('start', ascending = False).head()
# Bottom 5 starting salaries.

salaries_major.sort_values('start').head()
# Bar plot for starting salaries by major.

plt.figure(figsize = (10, 10))

sns.barplot(x = salaries_major.sort_values('start', ascending = False).start, # Put data in descending order by start salary.

            y = salaries_major.sort_values('start', ascending = False).major) # Makes the graph easier to interpret.

plt.title('Mean Starting Salaries\nby College Major', fontsize = 18)

plt.xlabel('Salary (USD)', fontsize = 14)

plt.ylabel('Major', fontsize = 14)

plt.grid(axis = 'x')
# Top 5 mid-career salaries.

salaries_major.sort_values('mid_career', ascending = False).head()
# Bottom 5 mid-career salaries.

salaries_major.sort_values('mid_career').head()
# Bar plot for mid-career salaries by major,

plt.figure(figsize = (10, 10))

sns.barplot(x = salaries_major.sort_values('mid_career', ascending = False).mid_career, # Put data in descending order by mid-career salary.

            y = salaries_major.sort_values('mid_career', ascending = False).major) # Makes graph easier to interpret.

plt.title('Mean Mid-Career Salaries\nby College Major', fontsize = 18)

plt.xlabel('Salary (USD)', fontsize = 14)

plt.ylabel('Major', fontsize = 14)

plt.grid(axis = 'x')
# First create a new df for starting salaries.

start = salaries_major.loc[:, ['major', 'start']].sort_values('start', ascending = False)

start.rename(columns = {'start': 'salary'}, inplace = True)



# Add salary classifier col.

classify = []

for i in range(len(start)):

    classify.append('Starting')

start['salary_type'] = classify



start.head()
# Then create a df for mid-career salaries.

mid = salaries_major.loc[:, ['major', 'mid_career']]

mid.rename(columns = {'mid_career': 'salary'}, inplace = True)



# Add salary classifier col.

classify = []

for i in range(len(mid)):

    classify.append('Mid-Career')

mid['salary_type'] = classify



mid.head()
# Combine the two dfs to form one big df with salaries classified as either starting or mid-career.

combined = pd.concat([start, mid]).reset_index()

combined
# Grouped bar plot showing mean starting salary and mean mid-career salary for each major.

plt.figure(figsize = (10, 10))

sns.barplot(x = 'salary', y = 'major', hue = 'salary_type', data = combined.sort_values(['salary_type', 'salary'], ascending = [True, False]))

plt.title('Mean Starting vs Mid-Career\nSalaries by College Major', fontsize = 18)

plt.xlabel('Salary (USD)', fontsize = 14)

plt.ylabel('Major', fontsize = 14)

plt.grid(axis = 'x')
# Top 5 percent salary growth.

salaries_major.sort_values('delta_start_mid', ascending = False).head()
# Bottom 5 percent salary growth.

salaries_major.sort_values('delta_start_mid').head()
# Bar plot showing salary growth for each major

plt.figure(figsize = (10, 10))

sns.barplot(x = salaries_major.sort_values('delta_start_mid', ascending = False).delta_start_mid,

           y = salaries_major.sort_values('delta_start_mid', ascending = False).major)

plt.title('Mean % Salary Growth\nby College Major', fontsize = 18)

plt.xlabel('Salary Growth Rate (%)', fontsize = 14)

plt.ylabel('Major', fontsize = 14)

plt.grid(axis = 'x')
# Preview main df again.

salaries_major.head()
# Create subest dfs, then concatenate them together into a combined df.

df_10 = salaries_major[['major', 'mid_10p']].rename(columns = {'mid_10p': 'salary'})

df_25 = salaries_major[['major', 'mid_25p']].rename(columns = {'mid_25p': 'salary'})

df_50 = salaries_major[['major', 'mid_career']].rename(columns = {'mid_career': 'salary'})

df_75 = salaries_major[['major', 'mid_75p']].rename(columns = {'mid_75p': 'salary'})

df_90 = salaries_major[['major', 'mid_90p']].rename(columns = {'mid_90p': 'salary'})



combined = pd.concat([df_10, df_25, df_50, df_75, df_90]).reset_index()

combined
# Create salary percentile classifier col and add it to the combined df.

classifiers = ['10th Percentile', '25th Percentile', '50th Percentile',

              '75th Percentile', '90th Percentile']



classify = []

indicator = 0

for i in range(len(combined)):

    classify.append(classifiers[indicator])

    if len(classify) % 50 == 0:

        indicator +=1



combined['percentile'] = classify

combined
# Scatter plot using subset df.

plt.figure(figsize = (10, 10))

sns.scatterplot(x = 'salary', y = 'major', hue = 'percentile', data = combined.sort_values(['percentile', 'salary'], ascending = [False, True]))

plt.title('Mid-Career Salary Percentiles\nby College Major', fontsize = 18)

plt.xlabel('Salary (USD)', fontsize = 14)

plt.ylabel('Major', fontsize = 14)

plt.grid()