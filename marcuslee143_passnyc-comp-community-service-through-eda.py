# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# data visualization, exploration
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = 'large'

# I reuse variable names quite often so the below magic line is for enabling autocomplete via tab
%config IPCompleter.greedy=True
sat_explorer = pd.read_csv('../input/D5 SHSAT Registrations and Testers.csv')
sat_explorer.shape
sat_explorer.columns
sat_explorer.head()
sat_copy = sat_explorer.copy() # I want to keep the original data clean
sat_copy.drop('DBN', axis=1, inplace=True) # dropping the internal database key
# Calculating the above ratios
num_registered = sat_copy['Number of students who registered for the SHSAT']
num_participated = sat_copy['Number of students who took the SHSAT']
enrollment = sat_copy['Enrollment on 10/31']

sat_copy['registration_ratio'] = num_registered / enrollment
sat_copy['participation_ratio'] = num_participated / enrollment
sat_copy['commitment_ratio'] = num_participated / num_registered
# Sort ratios by descending
sat_sorted = sat_copy.sort_values(by=['registration_ratio',
                                      'participation_ratio',
                                      'commitment_ratio'], ascending=True, axis=0)
# pandas has weird logic for rows vs columns
# I believe axis=0 indicates that we are sorting on the ROWS, as opposed to the columns, and then we pick the COLUMN
# LABELS that we want to sort the rows on. You're welcome, college students.
sat_nonzero = sat_sorted.loc[sat_sorted['Number of students who registered for the SHSAT'] > 0]
sat_nonzero.head()
#TODO: With this smaller dataset, examine geographic overlay for each ratio (3 total)
school_explorer = pd.read_csv('../input/2016 School Explorer.csv')
school_explorer_columns = list(school_explorer.columns)
school_explorer_columns
list(sat_explorer.columns)
# There's 161 columns in the 2016 Schools Dataset. No way we need all of them.
school_explorer_geo = school_explorer[
    ['School Name',
     'Latitude',
     'Longitude',
#      'Address (Full)',
     'Economic Need Index', # index for housing, health, and free lunch welfare
     'School Income Estimate',
     'Percent ELL', # exactly one of the factors we wanted
     'Average ELA Proficiency'] # English proficiency measurement, another good proxy for benchmarking English teaching needs
]
school_explorer_geo.head()
# Here, I'm grouping by school, summing registration and test-taking numbers, and recalculating the proportions
# so that I can join this information on the school_explorer data.
sat_pregroup = sat_sorted.drop(['registration_ratio',
                                 'participation_ratio',
                                 'commitment_ratio',
                                 'Year of SHST',
                                 'Grade level'], axis=1)\
.rename(columns={'School name':'School Name', # necessary for merging tables later
                 'Enrollment on 10/31':'Total enrollment',
                 'Number of students who registered for the SHSAT':'Total registration',
                 'Number of students who took the SHSAT':'Total participation'})
sat_grouped = sat_pregroup.groupby('School Name').sum().reset_index()
# Redo calculations
num_registered = sat_grouped['Total registration']
num_participated = sat_grouped['Total participation']
enrollment = sat_grouped['Total enrollment']
sat_grouped['registration_ratio'] = num_registered / enrollment
sat_grouped['participation_ratio'] = num_participated / enrollment
sat_grouped['commitment_ratio'] = num_participated / num_registered
sat_grouped
pd.options.mode.chained_assignment = None  # default='warn'; not applicable to my scenario, was getting annoying
# lowercase school names and strip punctuation
school_explorer_geo['School Name'] = school_explorer_geo['School Name'].str.lower().str.replace(r'[^\w\s]', '')
sat_grouped['School Name'] = sat_grouped['School Name'].str.lower().str.replace(r'[^\w\s]', '')
# Combine the tables by school name.
sat_merge_school = sat_grouped.merge(school_explorer_geo, on="School Name")
sat_merge_school.sort_values(['registration_ratio', 'Total registration']).head()
# clean Percent ELL
sat_merge_school['Percent ELL'] = sat_merge_school['Percent ELL'].str.replace('%', '').astype(float) / 100
sat_merge_school.head()
print("Number of schools in 2016 data: " + str(len(school_explorer_geo)))
print("Number of schools in SHSAT data: " + str(len(sat_explorer.groupby('School name'))))
fig, [ax0, ax1] = plt.subplots(1,2, figsize=(15,6))
sns.regplot(x="Total registration", y="Economic Need Index", data=sat_merge_school, ax=ax0)
sns.regplot(x="registration_ratio", y="Economic Need Index", data=sat_merge_school, ax=ax1)
ax1.set_xlim(0,1.5)
ax0.set_ylim(0.4,1.0)
ax1.set_ylim(0.4,1.0)
fig, [ax0, ax1] = plt.subplots(1,2, figsize=(15,6))
sns.regplot(x="Total registration", y="Percent ELL", data=sat_merge_school, ax=ax0)
sns.regplot(x="registration_ratio", y="Percent ELL", data=sat_merge_school, ax=ax1)
ax1.set_xlim(0,1.5)
ax1.set_ylim(0,1.0)
fig, [ax0, ax1] = plt.subplots(1,2, figsize=(15,6))
sns.regplot(x="Total registration", y="Average ELA Proficiency", data=sat_merge_school, ax=ax0)
sns.regplot(x="registration_ratio", y="Average ELA Proficiency", data=sat_merge_school, ax=ax1)
ax1.set_xlim(0,1.5)
ax1.set_ylim(0,6)
#TODO: after aggregate analysis, dive deeper into trends 
# for Grade level and Year of the SHSAT (especially for data over time)
#TODO: Revisit Grade granular data, especially "Limited English Proficient" and "Economically Disadvantaged"
#TODO: Gather more information to break apart "Economic Need Index" (housing, health/disability, free lunch)