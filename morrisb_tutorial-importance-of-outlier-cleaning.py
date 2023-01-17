# To do linear algebra
import numpy as np

# To store the data
import pandas as pd

# To create plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# To create nicer plots
import seaborn as sns
# Load the data
df = pd.read_csv('../input/survey_results_public.csv', low_memory=False)

# Store the dirty and create a clean dataset
dirty_df = df[df.Gender.isin(['Male', 'Female'])][['Gender', 'ConvertedSalary']].dropna()
clean_df = df[(df.ConvertedSalary<500000) & (df.Gender.isin(['Male', 'Female']))][['Gender', 'ConvertedSalary']].dropna()
# Create the figure and the subplot-sizes
fig = plt.figure(1, figsize=(15,5))
gridspec.GridSpec(2,4)

# Plot: Unclean distribution
ax = plt.subplot2grid((2,4), (0,0), rowspan=1, colspan=3)
sns.distplot(dirty_df.ConvertedSalary, ax=ax)
plt.title('Dirty ConvertedSalary Distribution')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('Frequency')

# Plot: Unclean ConvertedSalary
ax = plt.subplot2grid((2,4), (0,3))
sns.pointplot(data=dirty_df, x='Gender', y='ConvertedSalary', ax=ax)
plt.title('Dirty Data')
plt.xlabel('Gender')
plt.ylabel(' Mean ConvertedSalary')
plt.ylim([0, 105000])

# Plot: Clean distribution
ax = plt.subplot2grid((2,4), (1,0), rowspan=1, colspan=3)
sns.distplot(clean_df.ConvertedSalary, ax=ax)
plt.title('Clean ConvertedSalary Distribution')
plt.xlabel('ConvertedSalary in US-Dollar')
plt.ylabel('Frequency')

# Plot: Clean ConvertedSalary
ax = plt.subplot2grid((2,4), (1,3))
sns.pointplot(data=clean_df, x='Gender', y='ConvertedSalary', ax=ax)
plt.title('Clean Data')
plt.xlabel('Gender')
plt.ylabel('Mean ConvertedSalary')
plt.ylim([0, 105000])

# Show the plot
fig.tight_layout()
plt.show()
# Compute mean/median with dirty data
dirty_salary = df[df.Gender.isin(['Male', 'Female'])][['Gender', 'ConvertedSalary']].dropna().groupby('Gender').agg({'ConvertedSalary':['mean', 'median']})
dirty_salary.columns = dirty_salary.columns.get_level_values(1)
dirty_salary = dirty_salary.stack().reset_index().rename(columns={'level_1':'Type', 0:'Salary'})
dirty_salary['Status'] = 'dirty'

# Compute mean/median with clean data
clean_salary = df[(df.Gender.isin(['Male', 'Female'])) & (df.ConvertedSalary<500000)][['Gender', 'ConvertedSalary']].dropna().groupby('Gender').agg({'ConvertedSalary':['mean', 'median']})
clean_salary.columns = clean_salary.columns.get_level_values(1)
clean_salary = clean_salary.stack().reset_index().rename(columns={'level_1':'Type', 0:'Salary'})
clean_salary['Status'] = 'clean'

# Combine both datasets
salary = pd.concat([dirty_salary, clean_salary]).reset_index(drop=True)
# Compare the mean/median grouped by Status and Gender
sns.factorplot(data=salary, hue='Status', x='Type', y='Salary', col='Gender')
plt.show()
# Compute the difference between the dirty and the clean dataset
dirty_salary_values = salary[salary.Status=='dirty'].Salary.values
clean_salary_values = salary[salary.Status=='clean'].Salary.values

difference = salary.copy().head(4).drop('Status', axis=1).rename(columns={'Salary':'Decrease'})
difference['Decrease'] = (dirty_salary_values - clean_salary_values) / dirty_salary_values
# Compare thedifference in the mean/median grouped by Status and Gender
sns.barplot(data=difference, hue='Gender', x='Type', y='Decrease')
plt.title('Percental decrease between dirty and clean data')
plt.ylabel('Percental decrease')
plt.show()
