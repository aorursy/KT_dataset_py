import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
%matplotlib inline
# Data from the website www.data.gov
# The csv contains US surveillance data (causes of death) for various diseases including some data for COVID19
# Contains data for 2019 - 2020 for various genders age group and ethnicity.
df = pd.read_csv('../input/disease-data-from-the-cdc-for-2019-and-2020/Monthly_provisional_counts_of_deaths_by_age_group__sex__and_race_ethnicity_for_select_causes_of_death.csv')
df.info()
df.head(10)
df.sample(10)
# First step of data cleaning exercise - finding out how many missing values are in each column.
missing_values_count = df.isnull().sum()
missing_values_count[0:25]
# Getting a subset of the data to only include up to the column "COVID-19 (U071, Underlying Cause of Death)"
subset_cause_of_death = df.iloc[:, 0:20]
subset_cause_of_death.head()
# First step of possible data cleaning
# Find out what percentage of values are missing (i.e. have NaN)
missing_values_count2 = subset_cause_of_death.isnull().sum()
missing_values_count2[0:20]

total_cells2 = np.product(subset_cause_of_death.shape)
total_missing2 = missing_values_count2.sum()

(total_missing2/total_cells2) * 100
# Second step of data cleaning - replace null values
# Based on the notes on the meta data, and further clarification from a persons who works at cdc, 
# anything that is null may indicate a missing value or zero.
# Assumption for this analysis: anything left blank is 0.

cause_of_death_nulls_replaced = subset_cause_of_death.fillna(0)
cause_of_death_nulls_replaced.head()
# Limiting data to 2019
causeofdeath_2k19 = cause_of_death_nulls_replaced[cause_of_death_nulls_replaced['Date Of Death Year'] == 2019]
causeofdeath_2k19
# Bar plot showing the age groups with highest and lowest instances of malignant neoplasms.
plt.figure(figsize=(20,10))
sns.barplot(x='AgeGroup',y='Malignant neoplasms (C00-C97)',data=causeofdeath_2k19)
# selecting only data relating to diabetes
diabetesdata = cause_of_death_nulls_replaced[['Date Of Death Year','Date Of Death Month','Sex','Race/Ethnicity','AgeGroup','Diabetes mellitus (E10-E14)']]
diabetesdata
# Focusing on the Hispanic ethnic group
hispanicData = causeofdeath_2k19[causeofdeath_2k19['Race/Ethnicity'] == 'Hispanic']
hispanicData.head()
#Bar plot showing number of hispanic deaths from Diabetes per gender
plt.figure(figsize=(20,10))
sns.barplot(x='Sex',y='Diabetes mellitus (E10-E14)',data=hispanicData)
#Bar plot showing number of alzheimers deaths  per gender

plt.figure(figsize=(20,10))
sns.barplot(x='Sex',y='Alzheimer disease (G30)',data=hispanicData)
#Bar plot showing number of influenza and pneumonia deaths  per age group
plt.figure(figsize=(20,10))
sns.barplot(x='AgeGroup',y='Influenza and pneumonia (J09-J18)',data=hispanicData)
#Bar plot showing number of influenza and pneumonia deaths from Septicemia (A40-A41) per age group
plt.figure(figsize=(20,10))
sns.barplot(x='AgeGroup',y='Septicemia (A40-A41)',data=hispanicData)
# Limiting data to only cause of death by diabetes
diabetesdata
test = pd.pivot_table(hispanicData, index=['Date Of Death Year','Date Of Death Month'],values=['AgeGroup','Diabetes mellitus (E10-E14)','Influenza and pneumonia (J09-J18)','Chronic lower respiratory diseases (J40-J47)'],aggfunc=np.sum)
test

#pivotTable = diabetesdata.pivot('Date Of Death Month', 'Date Of Death Year', 'Diabetes mellitus (E10-E14)')
#Further limiting data to the year 2019
diabetesdata2k19 = diabetesdata[diabetesdata['Date Of Death Year'] == 2019]
diabetesdataWithGenderExcluded = diabetesdata2k19[['Date Of Death Month','AgeGroup','Diabetes mellitus (E10-E14)']]
diabetesdataWithGenderExcluded
diabetesdataWithGenderExcluded['AgeGroup'] = diabetesdataWithGenderExcluded['AgeGroup'].astype('str')

# Converts all numerical months to Text values.
import calendar
diabetesdataWithGenderExcluded['Date Of Death Month'] = df['Date Of Death Month'].apply(lambda x: calendar.month_abbr[x])
diabetesdataWithGenderExcluded.head()
# Pivoting data for use in heatmap
diabetesdataWithGenderExcludedpivot = diabetesdataWithGenderExcluded.pivot_table(values='Diabetes mellitus (E10-E14)',index='Date Of Death Month',columns='AgeGroup',aggfunc=np.sum)
diabetesdataWithGenderExcludedpivot
diabetesdataWithGenderExcluded.info()
# Heatmap to show the number of diabetes deaths per month per age group
plt.figure(figsize=(18,10))


ax = sns.heatmap(diabetesdataWithGenderExcludedpivot, cmap='BuPu',linecolor='white',linewidths=1, annot=True, fmt='.1f')

bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 13, rotation=45)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 13)