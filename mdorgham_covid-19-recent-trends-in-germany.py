# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
covid_de = pd.read_csv('/kaggle/input/covid19-tracking-germany/covid_de.csv')
covid_de['date'] = pd.to_datetime(covid_de['date'])
demographics = pd.read_csv('/kaggle/input/covid19-tracking-germany/demographics_de.csv')
covid_de.head()
print('Shape of dataset (num_rows, num_columns): ', covid_de.shape)
print('\n')
print('The date range of the data is: ', covid_de.date.min().strftime('%d.%m.%Y'), '-', covid_de.date.max().strftime('%d.%m.%Y'))
print('\n')
print('Dataset info.')
covid_de.info()
covid_de.describe()
print('Columns with missing values:')
missing_val_count_by_column = (covid_de.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
#fill age_group Na's with the most frequent value
most_freq_age = covid_de.age_group.value_counts().idxmax()
print('Most frequent age group: ', most_freq_age)
covid_de.age_group.fillna(most_freq_age, inplace=True)

#for gender missing values, we fill half of it with 'M' and half with 'F'
mask = covid_de.gender.isna() 
ind = covid_de.gender.loc[mask].sample(frac=0.5).index
covid_de.loc[ind, 'gender'] = 'M'
covid_de.gender.fillna('F', inplace=True)
demographics.head()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from matplotlib.gridspec import GridSpec


covid_by_date = covid_de.groupby('date').sum()
#covid_by_date.sort_index(ascending=False, inplace=True)
#covid_by_date

sns.set_style("whitegrid")
plt.figure(figsize=(18,9))
plt.title('Daily infected cases reported.')
p=sns.lineplot(data=covid_by_date['cases'])
plt.figure(figsize=(18,9))
plt.title('Daily deaths reported.')
p=sns.lineplot(data=covid_by_date['deaths'])
covid_by_state = covid_de.groupby('state').sum()
covid_by_state.sort_values('cases', ascending=False, inplace=True)

plt.figure(figsize=(14,10))
sns.barplot(covid_by_state.cases, covid_by_state.index, palette='Spectral');
plt.title("Cases by federal state")
plt.xlabel("No. of cases")
p=plt.ylabel("State")
mask = (covid_de['date'] > '2020-3-15') & (covid_de['date'] <= '2020-4-15')
covid_masked_old = covid_de.loc[mask]
mask = (covid_de['date'] > '2020-7-14') & (covid_de['date'] <= '2020-8-14')
covid_masked_recent = covid_de.loc[mask]

covid_by_state_old = covid_masked_old.groupby('state').sum()
covid_by_state_old.sort_values('cases', ascending=False, inplace=True)
covid_by_state_recent = covid_masked_recent.groupby('state').sum()
covid_by_state_recent.sort_values('cases', ascending=False, inplace=True)

plt.figure(figsize=(26,15))
the_grid = GridSpec(2, 2)
plt.subplot(the_grid[0, 0],  title='15 March - 15 April 2020')
sns.barplot(x=covid_by_state_old.cases, y=covid_by_state_old.index, palette='Spectral')
plt.subplot(the_grid[0, 1], title='14 July - 14 August 2020')
sns.barplot(x=covid_by_state_recent.cases, y=covid_by_state_recent.index, palette='Spectral')
p=plt.suptitle('Cases by federal state', fontsize=16)

states_population = demographics.groupby('state').sum().sort_values('population', ascending=False)

states_population
covid_old_ratio = covid_by_state_old.cases / states_population.loc[covid_by_state_old.index].population
covid_old_ratio.sort_values(ascending=False, inplace=True)

covid_recent_ratio = covid_by_state_recent.cases / states_population.loc[covid_by_state_recent.index].population
covid_recent_ratio.sort_values(ascending=False, inplace=True)


plt.figure(figsize=(26,15))
the_grid = GridSpec(2, 2)
plt.subplot(the_grid[0, 0],  title='15 March - 15 April 2020')
sns.barplot(x=covid_old_ratio, y=covid_old_ratio.index, palette='Spectral')
plt.subplot(the_grid[0, 1], title='14 July - 14 August 2020')
sns.barplot(x=covid_recent_ratio, y=covid_recent_ratio.index, palette='Spectral')
p=plt.suptitle('Cases to Population ratios by federal state', fontsize=16)

covid_by_age = covid_de.groupby('age_group').sum()
covid_by_age.sort_values('cases', ascending=False, inplace=True)

cmap = plt.get_cmap('Spectral')
colors = [cmap(i) for i in np.linspace(0, 1, 8)]

plt.figure(figsize=(8,8))
cases_pie = plt.pie(covid_by_age.cases, labels=covid_by_age.index, autopct='%1.1f%%', shadow=False, colors=colors)
deaths_ratio = covid_by_age.deaths / covid_by_age.cases * 100.0
covid_by_age['death_ratio'] = deaths_ratio
covid_by_age.sort_values('death_ratio', ascending=False, inplace=True)

plt.figure(figsize=(8,8))
death_pie = sns.barplot(y=covid_by_age.death_ratio, x=covid_by_age.index, palette='Spectral')
p=plt.xlabel('Age group')
p=plt.ylabel('Death ratio %')
covid_by_gender = covid_de.groupby('gender').sum()
covid_by_gender.sort_values('cases', ascending=False, inplace=True)

plt.figure(figsize=(16,20))
plt.subplot(the_grid[0, 0], aspect=1, title='Cases by gender')
cases_pie = plt.pie(covid_by_gender.cases, labels=covid_by_gender.index, autopct='%1.1f%%')


plt.subplot(the_grid[0, 1], aspect=1, title='Deaths by gender')
death_pie = plt.pie(covid_by_gender.deaths, labels=covid_by_gender.index, autopct='%1.1f%%')
covid_by_sg = covid_de.groupby(['age_group', 'gender'],as_index=False).sum()
#covid_by_sg

plt.figure(figsize=(16,8))
p = sns.barplot(y=covid_by_sg.deaths, x=covid_by_sg.age_group, hue=covid_by_sg.gender, data=covid_by_sg);
#plt.xticks(rotation=-45)
p=plt.title("Deaths by age group")
p=plt.ylabel("deaths")
p=plt.xlabel("Age group")