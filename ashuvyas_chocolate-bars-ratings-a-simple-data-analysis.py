# Import packages and setup the data directory

import os, math, itertools, matplotlib

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import matplotlib.pylab as pylab



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Setting up the plots font size and figure dpi

plt.rcParams.update({'font.size': 7})

dpi_set = 1200



# Read the file into pandas dataframe

main_path = '/kaggle/input/chocolate-bar-ratings'

main_data = pd.read_csv(main_path+"//flavors_of_cacao.csv")



# Cleaning the data

main_data.columns = main_data.columns.str.replace('\n', ' ').str.replace('\xa0', '')



emptyval = main_data['Bean Type'].values[0]

def empty_to_nan(feature):

    if feature == emptyval:

        return np.nan

    else:

        return feature



for onecol in main_data.columns:

    if main_data[onecol].dtype == 'O':

        main_data[onecol] = main_data[onecol].apply(lambda element: empty_to_nan(element))



# Converting rating and cocoa percentage strings to numbers

main_data['Cocoa Percent'] = main_data['Cocoa Percent'].apply(lambda row: row[:-1]).astype('float')



print(main_data.head(5))
# List of all columns 

features_list = list(main_data.columns)

print(features_list)
#### Plot Ratings

plot_df = main_data['Rating'].value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100

plot_df_data = list(plot_df)

plot_df_uniques = list(plot_df.keys())

plt.figure(dpi=dpi_set,figsize=(4,3))

plt.xlabel('Ratings')

plt.ylabel('Percent [%]')

plt.bar(plot_df_uniques,height=plot_df_data, width=0.17,linewidth=0,edgecolor='w')

plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)

plt.show()

#### Plot Review Date

plot_df = main_data['Review Date'].value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100

plot_df_data = list(plot_df)

plot_df_uniques = list(plot_df.keys())

plt.figure(dpi=dpi_set,figsize=(4,3))

plt.xlabel('Review Date')

plt.ylabel('Percent [%]')

plt.bar(plot_df_uniques,height=plot_df_data, width=0.3,linewidth=0,edgecolor='w')

plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)

plt.show()

#### Plot REF

plot_df = main_data['REF'].value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100

plot_df_data = list(plot_df)

plot_df_uniques = list(plot_df.keys())

plt.figure(dpi=dpi_set,figsize=(4,3))

plt.xlabel('Batch Reference No.')

plt.ylabel('Percent [%]')

plt.scatter(plot_df_uniques,plot_df_data, s=5)

plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)

plt.show()

#### Plot Cocoa Percent

plot_df = main_data['Cocoa Percent'].value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100

plot_df_data = list(plot_df)

plot_df_uniques = list(plot_df.keys())

plt.figure(dpi=dpi_set,figsize=(4,3))

plt.xticks(rotation=90)

plt.xlabel('Cocoa Percent [%]')

plt.ylabel('Percent [%]')

plt.scatter(plot_df_uniques,plot_df_data, s=7)

plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)

plt.show()

#### Plot Bean Type

plot_df = main_data['Bean Type'].fillna(value='Unavailable', method=None, axis=None, inplace=False, limit=None, downcast=None) 

plot_df = plot_df.value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100

plot_df_data = list(plot_df)

plot_df_uniques = list(plot_df.keys())

plt.figure(dpi=dpi_set,figsize=(7.875,3.375))

plt.xticks(rotation=90)

plt.xlabel('Bean Type')

plt.ylabel('Percent [%]')

plt.scatter(plot_df_uniques,plot_df_data, s=8)

plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)

plt.show()

#### Plot Company Location

plot_df = main_data['Company Location'].fillna(value='Unavailable', method=None, axis=None, inplace=False, limit=None, downcast=None) 

plot_df = plot_df.value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100

plot_df_data = list(plot_df)

plot_df_uniques = list(plot_df.keys())

plt.figure(dpi=dpi_set,figsize=(7.875,3.375))

plt.xticks(rotation=90)

plt.xlabel('Company Location')

plt.ylabel('Percent [%]')

plt.scatter(plot_df_uniques,plot_df_data, s=10)

plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)

plt.show()

#### Plot Broad Bean Origin

plot_df = main_data['Broad Bean Origin'].fillna(value='Unavailable', method=None, axis=None, inplace=False, limit=None, downcast=None) 

plot_df = plot_df.value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100

plot_df_data = list(plot_df)

plot_df_uniques = list(plot_df.keys())

plt.figure(dpi=dpi_set,figsize=(11.875,3.375))

plt.xticks(rotation=90)

plt.xlabel('Broad Bean Origin')

plt.ylabel('Percent [%]')

plt.scatter(plot_df_uniques,plot_df_data, s=10)

plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)

plt.show()

#### Plot Specific Bean Origin or Bar Name

plot_df = main_data['Specific Bean Origin or Bar Name'].fillna(value='Unavailable', method=None, axis=None, inplace=False, limit=None, downcast=None) 

plot_df = plot_df.value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100

plot_df_data = list(plot_df)

plot_df_uniques = list(plot_df.keys())

plt.figure(dpi=dpi_set,figsize=(7.875,3.375))

plt.xticks(rotation=90)

plt.xlabel('Specific Bean Origin or Bar Name')

plt.ylabel('Percent [%]')

plt.scatter(plot_df_uniques,plot_df_data, s=7)

plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)

plt.show()

#### Plot Company (Maker-if known)

plot_df = main_data['Company (Maker-if known)'].fillna(value='Unavailable', method=None, axis=None, inplace=False, limit=None, downcast=None) 

plot_df = plot_df.value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False)*100

plot_df_data = list(plot_df)

plot_df_uniques = list(plot_df.keys())

plt.figure(dpi=dpi_set,figsize=(7.875,3.375))

plt.xticks(rotation=90)

plt.xlabel('Company')

plt.ylabel('Percent [%]')

plt.scatter(plot_df_uniques,plot_df_data, s=5)

plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)

plt.show()

# set marker size based on the cocoa percent of a chocolate

markerprop = {'s': .1*main_data['Cocoa Percent']**1.7, 'alpha': .2, 'color': 'navy'}



# scatterplot

ax = sns.lmplot(x='Company Location', y='Rating', data=main_data, scatter_kws=markerprop, fit_reg=False, legend=True, aspect=2.5)



# add averaged rating

main_data_copy = main_data.copy().groupby(['Company Location'])['Rating'].mean()

plt.plot(main_data_copy, color='r', marker='o', alpha=.8, lw=3, label='Averaged Rating')



# # graph properties

ax.set_xticklabels(rotation=90)

plt.title('Rating vs. Company Location', fontsize=20)

plt.xlabel('Company Location', fontsize=15)

plt.ylabel('Rating', fontsize=15)

plt.margins(.03)

plt.show()
print("\nDone!!!\n")