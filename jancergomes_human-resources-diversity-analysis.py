import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

%matplotlib inline
hr_data = pd.read_csv('../input/human-resources-data-set/HRDataset_v13.csv')

hr_data.info()
# Lots of empty rows. Some cleaning now:

hr_data.dropna(how='all', inplace=True)
# A first look at the racial diversity situation - both in absolute and percentual figures:

hr_data.RaceDesc.name = 'Racial group'

display(hr_data.RaceDesc.value_counts(),

        hr_data.RaceDesc.value_counts(normalize=True) * 100)
hr_data['Non-white'] = (hr_data['RaceDesc'] != 'White')

fig = plt.figure(figsize=(6, 6)), sns.set_style('whitegrid')

# Ordering for better visualization

ordered_nw = hr_data.groupby('RecruitmentSource')['Non-white'].mean().reset_index().sort_values('Non-white', ascending=False)

ax = sns.barplot(data=ordered_nw, y='RecruitmentSource', x='Non-white', palette='Set2')

ax.set_xlabel('Non-white proportion'), ax.set_ylabel('Recruitment Source');
plt.figure(figsize=(6, 6))

ax = sns.boxplot(data=hr_data, x='RaceDesc', y='PayRate', palette='Set2')

ax.set_xticklabels(hr_data.RaceDesc.unique(), rotation=90)

sns.pointplot(data=hr_data, x='RaceDesc', y='PayRate', join=False, ci=None, ax=ax);

ax.set_ylabel('Hourly pay rate'); ax.set_title('Pay rate by racial group\n(mean indicated by blue points)');
g = sns.catplot(data=hr_data, x='RaceDesc', col_wrap=2,

                y='PayRate', col='Department', kind='box', palette='Set2')

g.set_xticklabels(hr_data.RaceDesc.unique(), rotation=90)

g.set_ylabels('Hourly pay rate')

g.fig.suptitle(

    'Pay rate ditribution by race group in each department', fontsize=16)

g.set_titles('{col_name}')

plt.subplots_adjust(top=0.95)
# Removing whitespace in "Data Analyst " (Thanks for noticing, Jordan!)

hr_data['Position'] = hr_data['Position'].str.strip()



ax = sns.scatterplot('PayRate', 'Position', 'RaceDesc', data=hr_data.loc[hr_data.Department=='IT/IS'],

                     palette='Set2', style='RaceDesc',size='RaceDesc', sizes=[40, 40, 40, 120, 40]);

ax.legend(bbox_to_anchor=(1, 1)).texts[0].set_text('Racial group'); ax.set_xlabel('Hourly pay rate'); 
# Filtering only rows that contain "two or more races" workers

position_rows = hr_data.Position.isin(['IT Support', 'IT Manager - DB'])

perf_indicators = ['RaceDesc','Position', 'PerformanceScore', 'SpecialProjectsCount', 'DaysLateLast30','EngagementSurvey']

it_is_lookup = hr_data.loc[position_rows, perf_indicators].sort_values(['Position','RaceDesc']).set_index('RaceDesc')

it_is_lookup
hr_data.loc[hr_data['Department'] == 'Executive Office'] 
# For clarity:

hr_data.replace({'Sex': {'F': 'Female', 'M ': 'Male'}}, inplace=True)

hr_data.Sex.name = 'Gender'

# Now, to an overview in gender distribution:

print(hr_data.Sex.value_counts(),'\n\n', (hr_data.Sex.value_counts(normalize=True) * 100), sep='')
# By department

g = sns.catplot(data=hr_data, x='Sex', col='Department',

                col_wrap=2, palette='Spectral', kind='count')

g.set_xlabels('Gender'), g.set_ylabels('# of employees')

g.fig.suptitle(

    'Employee count by gender in each department', fontsize=16)

g.set_titles('{col_name}')

plt.subplots_adjust(top=0.95)
GenderPay = hr_data.groupby('Sex')[['PayRate']]

display(GenderPay.agg(['mean', 'median']))
# Removing whitespace in 'Production':

hr_data['Department'] = hr_data['Department'].str.strip()

plt.figure(figsize=(8, 6))

ax = sns.pointplot(data=hr_data, x='Department', y='PayRate',

                   hue='Sex', palette='Spectral' , join=False)

ax.set_xticklabels(hr_data.Department.unique(), rotation=45,

                   horizontalalignment='right');

ax.legend()

ax.set_title("Average pay rate by department"), ax.set_ylabel(

    'Hourly pay rate');
ax = sns.stripplot('PayRate', 'Position', hue='Sex',

                   data=hr_data[hr_data.Department == 'Admin Offices'],

                  palette='Spectral')

ax.set_xlabel('Hourly pay rate'), ax.set_title('Pay rate in Production department')

ax.legend();
# We can make our own age column.

# But first, let's convert 'DOB' to datetime format, with a bit of chaining to keep just the date

from dateutil.relativedelta import relativedelta

hr_data['DOB'] = pd.to_datetime(hr_data['DOB']).dt.date.astype('datetime64')

# No employees were born after year 2000, so DOBs like 2068 should have 100 years removed:

hr_data.loc[hr_data.DOB.dt.year > 2000, 'DOB'] -= pd.DateOffset(years=100)

# Now, to getting the age:

hr_data['Age'] = pd.Series(dtype='int')

for ind, date in hr_data.DOB.iteritems():

    hr_data.loc[ind, 'Age'] = relativedelta(

        pd.to_datetime('today'), date).years

hr_data.Age = hr_data.Age.astype('int64')

#Some summary statistics

hr_data['Age'].apply(['min', 'median', 'max'])
ax = sns.distplot(hr_data['Age'], kde=False)
agegroups= pd.cut(hr_data.Age, [25,40,55,70])

agegroups.name = 'Age group'

agegroups.value_counts(normalize=True) *100
ordered_55 = hr_data.loc[hr_data.Age >= 55, 'RecruitmentSource'].value_counts()

ax = sns.barplot(x=ordered_55.values,y=ordered_55.index)

ax.set_ylabel('Recruitment source'), ax.set_xlabel('Employee count'), ax.set_xticks(range(6))

ax.set_title('Number of employees (ages 55+) by recruitment source');
g = sns.relplot('Age', 'PayRate', agegroups, hue_order=agegroups.cat.categories, col='Department',

 col_wrap=2, data=hr_data, palette='PuRd')

g.set_xlabels('Age'), g.set_ylabels('Hourly pay rate')

g.fig.suptitle('Age x Pay rate across departments', fontsize=16)

g.set_titles('{col_name}')

plt.subplots_adjust(top=0.95)
agegroups= pd.cut(hr_data.Age, [25,40,55,70])

ax = sns.boxplot('PayRate', 'Position', agegroups,

                   data=hr_data.loc[hr_data.Department=='Production'], palette='Dark2') ;

ax.set_xlabel('Hourly pay rate'); ax.set_title('Pay rate per position in Production');
# The department:

soft_engs = hr_data.Department == 'Software Engineering'

# Columns for analysis:

relevant_info = ['Employee_Name', 'Age', 'Position', 'PayRate',

                 'PerformanceScore', 'EmploymentStatus', 'EngagementSurvey']

# Now, we select them and look at some relevant columns:

hr_data.loc[soft_engs, relevant_info].sort_values('Age', ascending=False).set_index('Age')