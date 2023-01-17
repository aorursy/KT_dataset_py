# Importing the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

%matplotlib inline 
passport_data = pd.read_csv('../input/Passport_index.csv')
print(passport_data.head())
# Tracking the relative change of Global passport index ranking
passport_data['change_index'] = passport_data.rank_2015 - passport_data.rank_2018
passport_data = passport_data.sort_values('change_index')

plt.figure(figsize=(20,10))
f = sns.barplot(passport_data['Country_name'], passport_data['change_index'])
f.set_xticklabels(f.get_xticklabels(), rotation=90, fontsize=8);
# Tracking the relative change of Global passport index ranking
passport_data['change_index'] = passport_data.visa_free_score_2018 - passport_data.visa_free_score_2015
passport_data = passport_data.sort_values('change_index')

plt.figure(figsize=(20,10))
f = sns.barplot(passport_data['Country_name'], passport_data['change_index'])
f.set_xticklabels(f.get_xticklabels(), rotation=90, fontsize=8);
# Change in visa score with economic state
# Mention preprocessing

passport_data = passport_data.sort_values('rank_2018')
eco_status = pd.read_csv("../input/Country_status.csv", encoding='latin-1')
merge_data = passport_data.merge(eco_status, how = 'inner', left_on='Country_name', right_on='TableName')

plt.figure(figsize=(20,10))
f = sns.barplot(merge_data['IncomeGroup'],merge_data['rank_2018'])
f.set_xticklabels(f.get_xticklabels(), fontsize=12);

# Change in visa score with economic state

migrant_population = pd.read_csv("../input/migrant_population.csv", encoding='latin-1')
migrant_population['change_pop'] = migrant_population['2016'] - migrant_population['2013']
migrant_population = migrant_population[['Country Code', 'Country Name', 'change_pop']]
merge_data2 = merge_data.merge(migrant_population, how = 'inner', on='Country Code').sort_values('change_pop')
# 
migrant_population = pd.read_csv("../input/migrant_population.csv", encoding='latin-1')
migrant_population = migrant_population[['Country Code', 'Country Name', '2010','2011','2012','2013','2014','2015','2016']]
migrant_population['average_pop'] = migrant_population.mean(axis=1)
migrant_population = migrant_population[['Country Code', 'Country Name', 'average_pop']]
merge_data2 = merge_data.merge(migrant_population, how = 'inner', on='Country Code').sort_values('average_pop')

plt.figure(figsize=(20,30))
ax = sns.lmplot('rank_2018', 'average_pop', hue="IncomeGroup", data= merge_data2)
ax.set(yscale="log");
