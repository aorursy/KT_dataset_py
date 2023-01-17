# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import math

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))
# Load the .csv files into DataFrames.

child_mortality = pd.read_csv('../input/gapminder-dataset/child_mortality_0_5_year_olds_dying_per_1000_born.csv')

children_per_woman = pd.read_csv('../input/gapminder-dataset/children_per_woman_total_fertility.csv')

co2_emissions = pd.read_csv('../input/gapminder-dataset/co2_emissions_tonnes_per_person.csv')

income = pd.read_csv('../input/gapminder-dataset/income_per_person_gdppercapita_ppp_inflation_adjusted.csv')

life_expectancy = pd.read_csv('../input/gapminder-dataset/life_expectancy_years.csv')

population = pd.read_csv('../input/gapminder-dataset/population_total.csv')

country_info = pd.read_csv('../input/gapminder-dataset/country_info.csv')
# Creates a list country names.

countries = population['country'].tolist()
# Set the index of each dataframe to the 'country' column.

child_mortality = child_mortality.set_index('country')

children_per_woman = children_per_woman.set_index('country')

co2_emissions = co2_emissions.set_index('country')

income = income.set_index('country')

life_expectancy = life_expectancy.set_index('country')

population = population.set_index('country')

country_info = country_info.set_index('name')
# Create the dataframes for analysis and give them countries as indexes to match the above dataframes.

stats_1967 = pd.DataFrame(index=countries)

stats_2017 = pd.DataFrame(index=countries)
# Populate the 1967 dataframe with the chosen statistics.

stats_1967['child_mortality'] = child_mortality['1967']

stats_1967['children_per_woman'] = children_per_woman['1967']

stats_1967['co2_emissions'] = co2_emissions['1967']

stats_1967['income'] = income['1967']

stats_1967['life_expectancy'] = life_expectancy['1967']

stats_1967['population'] = population['1967']

stats_1967.head()
# Check data types.

stats_1967.dtypes
# Check for missing data.

stats_1967.isnull().sum()
# Impute missing values using a median strategy. 

myimputer = SimpleImputer(strategy='median')

stats_1967_imp = pd.DataFrame(myimputer.fit_transform(stats_1967), index=countries)

stats_1967_imp.columns = stats_1967.columns

stats_1967_imp.head()
# Re-check missing values. Should return 0.

stats_1967_imp.isnull().sum()
# Populate the 1967 dataframe with the chosen statistics.

stats_2017['child_mortality'] = child_mortality['2017']

stats_2017['children_per_woman'] = children_per_woman['2017']

stats_2017['co2_emissions'] = co2_emissions['2017']

stats_2017['income'] = income['2017']

stats_2017['life_expectancy'] = life_expectancy['2017']

stats_2017['population'] = population['2017']

stats_2017.head()
# Check data types.

stats_2017.dtypes
# Check for missing values.

stats_2017.isnull().sum()
# Impute missing values using a median strategy.

stats_2017_imp = pd.DataFrame(myimputer.fit_transform(stats_2017), index=countries)

stats_2017_imp.columns = stats_2017.columns

stats_2017_imp.head()
# Re-check for missing values. Should return 0.

stats_2017_imp.isnull().sum()
# Add country region, which will be used for visualization customization. 

stats_1967_imp['region'] = country_info['four_regions']

stats_2017_imp['region'] = country_info['four_regions']
# After identifying two missing region values, data exploration outside of this notebook led to the following adjustments.

stats_1967_imp.loc['Eswatini', 'region'] = 'africa'

stats_1967_imp.loc['North Macedonia', 'region'] = 'europe'

stats_2017_imp.loc['Eswatini', 'region'] = 'africa'

stats_2017_imp.loc['North Macedonia', 'region'] = 'europe'
# Shortening the dataframe name for ease of use.

stats1967 = stats_1967_imp

stats2017 = stats_2017_imp
# Adding a country column for labeling within visualizations.

stats1967['country'] = countries

stats2017['country'] = countries
# Creating a column of scaled population for use in visualization customization. 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

stats1967['scaled pop'] = scaler.fit_transform(stats1967[['population']])

stats2017['scaled pop'] = scaler.fit_transform(stats2017[['population']])
# In order to recreate some visualizations, child_mortality had to be transformed to child_survival_percent, done below.

stats1967['child_survival_percent'] = (1 - (stats1967['child_mortality'] / 1000)) * 100

stats2017['child_survival_percent'] = (1 - (stats2017['child_mortality'] / 1000)) * 100
# The following code plots the distribution of each column in both 1967 and 2017.

# These plots serve as a check of the data, as well as a way to examine how many of the key metrics have changed over the past 50 years.

fig1, axes = plt.subplots(3, 2, figsize=(20, 10))

fig1.suptitle('Visualization of Data Distributions, 1967 (Blue) and 2017 (Orange)')

sns.kdeplot(ax=axes[0,0], data=stats1967['child_mortality'])

sns.kdeplot(ax=axes[0,0], data=stats2017['child_mortality'])

sns.kdeplot(ax=axes[0,1], data=stats1967['children_per_woman'])

sns.kdeplot(ax=axes[0,1], data=stats2017['children_per_woman'])

sns.kdeplot(ax=axes[1,0], data=stats1967['life_expectancy'])

sns.kdeplot(ax=axes[1,0], data=stats2017['life_expectancy'])

sns.kdeplot(ax=axes[1,1], data=stats1967['income'])

sns.kdeplot(ax=axes[1,1], data=stats2017['income'])

sns.kdeplot(ax=axes[2,0], data=stats1967['population'])

sns.kdeplot(ax=axes[2,0], data=stats2017['population'])

sns.kdeplot(ax=axes[2,1], data=stats1967['co2_emissions'])

sns.kdeplot(ax=axes[2,1], data=stats2017['co2_emissions'])
# Creates a function for labeling data points within the scatter plots. Labeling each point overwhelms the plot, thus I 've created a list

# of a handful of 'key' countries to label. These may be altered to suit individual interests.

key_countries = ['China', 'India', 'United States', 'Germany', 'Iran', 'Afghanistan', 'Japan', 'South Korea', 'South Sudan', 'Congo, Dem. Rep.']

def label_point(x, y, value, ax):

    a = pd.concat({'x': x, 'y': y, 'value': value}, axis=1)

    for i, point in a.iterrows():

        if i in key_countries:

            ax.text(point['x']+0.1, point['y']+0.1, str(point['value']))

        else:

            pass
#plt.figure(figsize=(9,6))

#splot1 = sns.scatterplot(data=stats2017, x='income', y='life_expectancy', hue='region', size='scaled pop',

#                        sizes=(25, 2500), legend=False)

#splot1.set(xscale='log', xlim=(250, 200000), ylim=(35, 90), xlabel='INCOME', ylabel='LIFE EXPECTANCY', title='Income vs. Life Expectancy in 2017')

#sns.despine()

#label_point(stats2017['income'], stats2017['life_expectancy'], stats2017['country'], splot1)
#plt.figure(figsize=(9,6))

#splot2 = sns.scatterplot(data=stats1967, x='income', y='life_expectancy', hue='region', size='scaled pop',

#                        sizes=(25, 2500), legend=False)

#splot2.set(xscale='log', xlim=(250, 200000), ylim=(35, 90), xlabel='INCOME', ylabel='LIFE EXPECTANCY', title='Income vs. Life Expectancy in 1967')

#sns.despine()
#plt.figure(figsize=(9,6))

#splot3 = sns.scatterplot(data=stats2017, x='children_per_woman', y='child_survival_percent', hue='region', size='scaled pop',

#                        sizes=(25, 2500), legend=False)

#splot3.set(xscale='linear', xlim=(9, -1), ylim=(55, 105), xlabel='BABIES PER WOMAN', ylabel=' PERCENT CHILDREN WHO SURVIVE UNTIL AGE 5', title='Chilren Born vs. Child Mortality in 2017')

#sns.despine()
#plt.figure(figsize=(9,6))

#splot4 = sns.scatterplot(data=stats1967, x='children_per_woman', y='child_survival_percent', hue='region', size='scaled pop',

#                        sizes=(25, 2500), legend=False)

#splot4.set(xscale='linear', xlim=(9, -1), ylim=(55, 105), xlabel='BABIES PER WOMAN', ylabel=' PERCENT CHILDREN WHO SURVIVE UNTIL AGE 5', title='Chilren Born vs. Child Mortality in 1967')

#sns.despine()
#plt.figure(figsize=(9,6))

#splot5 = sns.scatterplot(data=stats2017, x='income', y='co2_emissions', hue='region', size='scaled pop',

#                        sizes=(25, 2500), legend=False)

#splot5.set(xscale='log', xlabel='INCOME', ylabel='CO2 EMISSIONS', title='Income vs CO2 Emissions in 2017')

#sns.despine()
#plt.figure(figsize=(9,6))

#splot6 = sns.scatterplot(data=stats1967, x='income', y='co2_emissions', hue='region', size='scaled pop',

#                        sizes=(25, 2500), legend=False)

#splot6.set(xscale='log', xlabel='INCOME', ylabel='CO2 EMISSIONS', title='Income vs CO2 Emissions in 1967')

#sns.despine()
# The code below creates a figure with six plots, each of the key plots (3) mentioned earlier for each of the years of interest (2).

fig2, axes = plt.subplots(3, 2, figsize=(20, 20))

fig2.suptitle('KEY VISUALIZATIONS')

sns.scatterplot(ax=axes[0,0], data=stats1967, x='income', y='life_expectancy', hue='region', size='scaled pop',sizes=(25, 2500), legend=False)

axes[0,0].set(xscale='log', xlim=(250, 200000), ylim=(35, 90), xlabel='INCOME', ylabel='LIFE EXPECTANCY', title='Income vs. Life Expectancy in 1967')

sns.scatterplot(ax=axes[0,1], data=stats2017, x='income', y='life_expectancy', hue='region', size='scaled pop',sizes=(25, 2500), legend=False)

axes[0, 1].set(xscale='log', xlim=(250, 200000), ylim=(35, 90), xlabel='INCOME', ylabel='LIFE EXPECTANCY', title='Income vs. Life Expectancy in 2017')

sns.scatterplot(ax=axes[1,0], data=stats1967, x='children_per_woman', y='child_survival_percent', hue='region', size='scaled pop',sizes=(25, 2500), legend=False)

axes[1,0].set(xscale='linear', xlim=(9, -1), ylim=(55, 105), xlabel='BABIES PER WOMAN', ylabel=' PERCENT CHILDREN WHO SURVIVE UNTIL AGE 5', title='Chilren Born vs. Child Mortality in 1967')

sns.scatterplot(ax=axes[1,1], data=stats2017, x='children_per_woman', y='child_survival_percent', hue='region', size='scaled pop',sizes=(25, 2500), legend=False)

axes[1,1].set(xscale='linear', xlim=(9, -1), ylim=(55, 105), xlabel='BABIES PER WOMAN', ylabel=' PERCENT CHILDREN WHO SURVIVE UNTIL AGE 5', title='Chilren Born vs. Child Mortality in 2017')

sns.scatterplot(ax=axes[2,0], data=stats1967, x='income', y='co2_emissions', hue='region', size='scaled pop', sizes=(25, 2500), legend=False)

axes[2,0].set(xscale='log', xlabel='INCOME', ylabel='CO2 EMISSIONS', title='Income vs CO2 Emissions in 1967')

sns.scatterplot(ax=axes[2,1], data=stats2017, x='income', y='co2_emissions', hue='region', size='scaled pop', sizes=(25, 2500), legend=False)

axes[2,1].set(xscale='log', xlabel='INCOME', ylabel='CO2 EMISSIONS', title='Income vs CO2 Emissions in 1967')

label_point(stats1967['income'], stats1967['life_expectancy'], stats1967['country'], axes[0,0])

label_point(stats2017['income'], stats2017['life_expectancy'], stats2017['country'], axes[0,1])

label_point(stats1967['children_per_woman'], stats1967['child_survival_percent'], stats1967['country'], axes[1,0])

label_point(stats2017['children_per_woman'], stats2017['child_survival_percent'], stats2017['country'], axes[1,1])

label_point(stats1967['income'], stats1967['co2_emissions'], stats1967['country'], axes[2,0])

label_point(stats2017['income'], stats2017['co2_emissions'], stats2017['country'], axes[2,1])

sns.despine()
# Import the model selected for use in this analysis.

from sklearn.cluster import KMeans
# Select a subset of columns (the five key metrics of the dataset) to input into the model.

columns_for_modeling = ['child_mortality', 'children_per_woman', 'co2_emissions', 'income', 'life_expectancy'] # Removed population after first pass.

data2017 = stats2017[columns_for_modeling]

data1967 = stats1967[columns_for_modeling]

# Standardize the data for use in the KMeans model.

data2017s = pd.DataFrame(scaler.fit_transform(data2017))

data1967s = pd.DataFrame(scaler.fit_transform(data1967))

data2017s.head()
# This code evaluates inertia for a range of n_clusters.

inertia2017 = []

for clusters in range(1, 10):

    model = KMeans(n_clusters=clusters)

    model.fit(data2017s)

    inertia2017.append(model.inertia_)

print(inertia2017)
# Plot the change in inertia against change in n_clusters.

sns.lineplot(x=range(1,10), y=inertia2017, color='black')
# This code evaluates inertia for a range of n_clusters.

inertia1967 = []

for clusters in range(1, 10):

    model = KMeans(n_clusters=clusters)

    model.fit(data1967s)

    inertia1967.append(model.inertia_)

print(inertia1967)
# Plot the change in inertia against change in n_clusters.

sns.lineplot(x=range(1,10), y=inertia1967, color='black')
# Create models with varying n_clusters.

model2 = KMeans(n_clusters=2)

model3 = KMeans(n_clusters=3)

model4 = KMeans(n_clusters=4)

model5 = KMeans(n_clusters=5)
# This cell fits and predicts the model, puts the predictions in a dataframe, indexes the dataframe, and renames the column.

y2_17 = pd.DataFrame(model2.fit_predict(data2017s), index=countries, columns=['groups_2'])

y2_67 = pd.DataFrame(model2.fit_predict(data1967s), index=countries, columns=['groups_2'])

y3_17 = pd.DataFrame(model3.fit_predict(data2017s), index=countries, columns=['groups_3'])

y3_67 = pd.DataFrame(model3.fit_predict(data1967s), index=countries, columns=['groups_3'])

y4_17 = pd.DataFrame(model4.fit_predict(data2017s), index=countries, columns=['groups_4'])

y4_67 = pd.DataFrame(model4.fit_predict(data1967s), index=countries, columns=['groups_4'])

y5_17 = pd.DataFrame(model5.fit_predict(data2017s), index=countries, columns=['groups_5'])

y5_67 = pd.DataFrame(model5.fit_predict(data1967s), index=countries, columns=['groups_5'])
# Adds the predictions from each model to its corresponding dataframe (1967 or 2017).

stats2017['groups_2'] = y2_17['groups_2']

stats2017['groups_3'] = y3_17['groups_3']

stats2017['groups_4'] = y4_17['groups_4']

stats2017['groups_5'] = y5_17['groups_5']

stats1967['groups_2'] = y2_67['groups_2']

stats1967['groups_3'] = y3_67['groups_3']

stats1967['groups_4'] = y4_67['groups_4']

stats1967['groups_5'] = y5_67['groups_5']
# This code allows us to inspect the number of countries in each group.

print(stats2017['groups_2'].value_counts())

print(stats2017['groups_3'].value_counts())

print(stats2017['groups_4'].value_counts())

print(stats2017['groups_5'].value_counts())

print(stats1967['groups_2'].value_counts())

print(stats1967['groups_3'].value_counts())

print(stats1967['groups_4'].value_counts())

print(stats1967['groups_5'].value_counts())
# For plotting purposes, I've created an income_per_day column.

# A reminder, the income column is GDP/person.

stats2017['income_per_day'] = stats_2017['income']/365

stats1967['income_per_day'] = stats_1967['income']/365
#fig3, axes = plt.subplots(3, 2, figsize=(20, 20)

#fig3.suptitle('KEY VISUALIZATIONS')

#sns.scatterplot(ax=axes[0,0], data=stats1967, x='income', y='life_expectancy', hue='groups_2', size='scaled pop',sizes=(25, 2500), legend=False)

#axes[0,0].set(xscale='log', xlim=(250, 200000), ylim=(35, 90), xlabel='INCOME', ylabel='LIFE EXPECTANCY', title='Income vs. Life Expectancy in 1967')

#axes[0,0].vlines(x=[800, 3000, 12000, 45000], ymin=35, ymax=90, linewidth=1, color='black', linestyles='dotted')

#sns.scatterplot(ax=axes[0,1], data=stats2017, x='income', y='life_expectancy', hue='groups_5', size='scaled pop',sizes=(25, 2500), legend=False)

#axes[0,1].set(xscale='log', xlim=(250, 200000), ylim=(35, 90), xlabel='INCOME', ylabel='LIFE EXPECTANCY', title='Income vs. Life Expectancy in 2017')

#axes[0,1].vlines(x=[800, 3000, 12000, 45000], ymin=35, ymax=90, linewidth=1, color='black', linestyles='dotted')

#sns.scatterplot(ax=axes[1,0], data=stats1967, x='children_per_woman', y='child_survival_percent', hue='groups_2', size='scaled pop',sizes=(25, 2500), legend=False)

#axes[1,0].set(xscale='linear', xlim=(9, -1), ylim=(55, 105), xlabel='BABIES PER WOMAN', ylabel=' PERCENT CHILDREN WHO SURVIVE UNTIL AGE 5', title='Chilren Born vs. Child Mortality in 1967')

#sns.scatterplot(ax=axes[1,1], data=stats2017, x='children_per_woman', y='child_survival_percent', hue='groups_5', size='scaled pop',sizes=(25, 2500), legend=False)

#axes[1,1].set(xscale='linear', xlim=(9, -1), ylim=(55, 105), xlabel='BABIES PER WOMAN', ylabel=' PERCENT CHILDREN WHO SURVIVE UNTIL AGE 5', title='Chilren Born vs. Child Mortality in 2017')

#sns.scatterplot(ax=axes[2,0], data=stats1967, x='income', y='co2_emissions', hue='groups_2', size='scaled pop', sizes=(25, 2500), legend=False)

#axes[2,0].set(xscale='log', xlim=(250, 200000), ylim=(-5, 80), xlabel='INCOME', ylabel='CO2 EMISSIONS', title='Income vs CO2 Emissions in 1967')

#axes[2,0].vlines(x=[800, 3000, 12000, 45000], ymin=-5, ymax=80, linewidth=1, color='black', linestyles='dotted')

#sns.scatterplot(ax=axes[2,1], data=stats2017, x='income', y='co2_emissions', hue='groups_5', size='scaled pop', sizes=(25, 2500), legend=False)

#axes[2,1].set(xscale='log', xlim=(250, 200000), ylim=(-5, 80), xlabel='INCOME', ylabel='CO2 EMISSIONS', title='Income vs CO2 Emissions in 1967')

#axes[2,1].vlines(x=[800, 3000, 12000, 45000], ymin=-5, ymax=80, linewidth=1, color='black', linestyles='dotted')

#label_point(stats1967['income'], stats1967['life_expectancy'], stats1967['country'], axes[0,0])

#label_point(stats2017['income'], stats2017['life_expectancy'], stats2017['country'], axes[0,1])

#label_point(stats1967['children_per_woman'], stats1967['child_survival_percent'], stats1967['country'], axes[1,0])

#label_point(stats2017['children_per_woman'], stats2017['child_survival_percent'], stats2017['country'], axes[1,1])

#label_point(stats1967['income'], stats1967['co2_emissions'], stats1967['country'], axes[2,0])

#label_point(stats2017['income'], stats2017['co2_emissions'], stats2017['country'], axes[2,1])

#sns.despine()
# Here I re-create the plots from above. A few adjustments have been made. Income is now Income Per Day. I've also added vertical lines

# to represent the divisions between the income levels as defined by Factfulness ($2, $8, $32 per day).

fig4, axes = plt.subplots(3, 2, figsize=(20, 20))

fig4.suptitle('KEY VISUALIZATIONS')

sns.scatterplot(ax=axes[0,0], data=stats1967, x='income_per_day', y='life_expectancy', hue='groups_2', size='scaled pop',sizes=(25, 2500), legend=False)

axes[0,0].set(xscale='log', xlim=(1, 500), ylim=(35, 90), xlabel='INCOME PER DAY', ylabel='LIFE EXPECTANCY', title='Income vs. Life Expectancy in 1967')

axes[0,0].vlines(x=[2, 8, 32], ymin=35, ymax=90, linewidth=1, color='black', linestyles='dotted')

sns.scatterplot(ax=axes[0,1], data=stats2017, x='income_per_day', y='life_expectancy', hue='groups_4', size='scaled pop',sizes=(25, 2500), legend=False)

axes[0,1].set(xscale='log', xlim=(1, 500), ylim=(35, 90), xlabel='INCOME PER DAY', ylabel='LIFE EXPECTANCY', title='Income vs. Life Expectancy in 2017')

axes[0,1].vlines(x=[2, 8, 32], ymin=35, ymax=90, linewidth=1, color='black', linestyles='dotted')

sns.scatterplot(ax=axes[1,0], data=stats1967, x='children_per_woman', y='child_survival_percent', hue='groups_2', size='scaled pop',sizes=(25, 2500), legend=False)

axes[1,0].set(xscale='linear', xlim=(9, -1), ylim=(55, 105), xlabel='BABIES PER WOMAN', ylabel=' PERCENT CHILDREN WHO SURVIVE UNTIL AGE 5', title='Chilren Born vs. Child Mortality in 1967')

sns.scatterplot(ax=axes[1,1], data=stats2017, x='children_per_woman', y='child_survival_percent', hue='groups_4', size='scaled pop',sizes=(25, 2500), legend=False)

axes[1,1].set(xscale='linear', xlim=(9, -1), ylim=(55, 105), xlabel='BABIES PER WOMAN', ylabel=' PERCENT CHILDREN WHO SURVIVE UNTIL AGE 5', title='Chilren Born vs. Child Mortality in 2017')

sns.scatterplot(ax=axes[2,0], data=stats1967, x='income_per_day', y='co2_emissions', hue='groups_2', size='scaled pop', sizes=(25, 2500), legend=False)

axes[2,0].set(xscale='log', xlim=(1, 500), ylim=(-5, 80), xlabel='INCOME PER DAY', ylabel='CO2 EMISSIONS', title='Income vs CO2 Emissions in 1967')

axes[2,0].vlines(x=[2, 8, 32], ymin=-5, ymax=80, linewidth=1, color='black', linestyles='dotted')

sns.scatterplot(ax=axes[2,1], data=stats2017, x='income_per_day', y='co2_emissions', hue='groups_4', size='scaled pop', sizes=(25, 2500), legend=False)

axes[2,1].set(xscale='log', xlim=(1, 500), ylim=(-5, 80), xlabel='INCOME PER DAY', ylabel='CO2 EMISSIONS', title='Income vs CO2 Emissions in 1967')

axes[2,1].vlines(x=[2, 8, 32], ymin=-5, ymax=80, linewidth=1, color='black', linestyles='dotted')

label_point(stats1967['income_per_day'], stats1967['life_expectancy'], stats1967['country'], axes[0,0])

label_point(stats2017['income_per_day'], stats2017['life_expectancy'], stats2017['country'], axes[0,1])

label_point(stats1967['children_per_woman'], stats1967['child_survival_percent'], stats1967['country'], axes[1,0])

label_point(stats2017['children_per_woman'], stats2017['child_survival_percent'], stats2017['country'], axes[1,1])

label_point(stats1967['income_per_day'], stats1967['co2_emissions'], stats1967['country'], axes[2,0])

label_point(stats2017['income_per_day'], stats2017['co2_emissions'], stats2017['country'], axes[2,1])

sns.despine()
fig5, axes = plt.subplots(3, 2, figsize=(20, 20))

fig5.suptitle('KEY VISUALIZATIONS')

sns.scatterplot(ax=axes[0,0], data=stats1967, x='income_per_day', y='life_expectancy', hue='groups_4', size='scaled pop',sizes=(25, 2500), legend=False)

axes[0,0].set(xscale='log', xlim=(1, 500), ylim=(35, 90), xlabel='INCOME PER DAY', ylabel='LIFE EXPECTANCY', title='Income vs. Life Expectancy in 1967')

axes[0,0].vlines(x=[4, 16, 64, 128], ymin=35, ymax=90, linewidth=1, color='black', linestyles='dotted')

sns.scatterplot(ax=axes[0,1], data=stats2017, x='income_per_day', y='life_expectancy', hue='groups_5', size='scaled pop',sizes=(25, 2500), legend=False)

axes[0,1].set(xscale='log', xlim=(1, 500), ylim=(35, 90), xlabel='INCOME PER DAY', ylabel='LIFE EXPECTANCY', title='Income vs. Life Expectancy in 2017')

axes[0,1].vlines(x=[4, 16, 64, 128], ymin=35, ymax=90, linewidth=1, color='black', linestyles='dotted')

sns.scatterplot(ax=axes[1,0], data=stats1967, x='children_per_woman', y='child_survival_percent', hue='groups_4', size='scaled pop',sizes=(25, 2500), legend=False)

axes[1,0].set(xscale='linear', xlim=(9, -1), ylim=(55, 105), xlabel='BABIES PER WOMAN', ylabel=' PERCENT CHILDREN WHO SURVIVE UNTIL AGE 5', title='Chilren Born vs. Child Mortality in 1967')

sns.scatterplot(ax=axes[1,1], data=stats2017, x='children_per_woman', y='child_survival_percent', hue='groups_5', size='scaled pop',sizes=(25, 2500), legend=False)

axes[1,1].set(xscale='linear', xlim=(9, -1), ylim=(55, 105), xlabel='BABIES PER WOMAN', ylabel=' PERCENT CHILDREN WHO SURVIVE UNTIL AGE 5', title='Chilren Born vs. Child Mortality in 2017')

sns.scatterplot(ax=axes[2,0], data=stats1967, x='income_per_day', y='co2_emissions', hue='groups_4', size='scaled pop', sizes=(25, 2500), legend=False)

axes[2,0].set(xscale='log', xlim=(1, 500), ylim=(-5, 80), xlabel='INCOME PER DAY', ylabel='CO2 EMISSIONS', title='Income vs CO2 Emissions in 1967')

axes[2,0].vlines(x=[4, 16, 64, 128], ymin=-5, ymax=80, linewidth=1, color='black', linestyles='dotted')

sns.scatterplot(ax=axes[2,1], data=stats2017, x='income_per_day', y='co2_emissions', hue='groups_5', size='scaled pop', sizes=(25, 2500), legend=False)

axes[2,1].set(xscale='log', xlim=(1, 500), ylim=(-5, 80), xlabel='INCOME PER DAY', ylabel='CO2 EMISSIONS', title='Income vs CO2 Emissions in 1967')

axes[2,1].vlines(x=[4, 16, 64, 128], ymin=-5, ymax=80, linewidth=1, color='black', linestyles='dotted')

label_point(stats1967['income_per_day'], stats1967['life_expectancy'], stats1967['country'], axes[0,0])

label_point(stats2017['income_per_day'], stats2017['life_expectancy'], stats2017['country'], axes[0,1])

label_point(stats1967['children_per_woman'], stats1967['child_survival_percent'], stats1967['country'], axes[1,0])

label_point(stats2017['children_per_woman'], stats2017['child_survival_percent'], stats2017['country'], axes[1,1])

label_point(stats1967['income_per_day'], stats1967['co2_emissions'], stats1967['country'], axes[2,0])

label_point(stats2017['income_per_day'], stats2017['co2_emissions'], stats2017['country'], axes[2,1])

sns.despine()